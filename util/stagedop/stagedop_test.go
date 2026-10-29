// Copyright (c) Tailscale Inc & contributors
// SPDX-License-Identifier: BSD-3-Clause

package stagedop

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// plan builds named stages that record each execution. failNext maps a stage
// name to the errors its next invocations should return, in order; when the
// list is exhausted the stage succeeds.
type recorder struct {
	mu          sync.Mutex
	calls       map[string]int
	active      int
	maxParallel int
	fail        map[string][]error
	order       []string
}

func newRecorder() *recorder {
	return &recorder{calls: map[string]int{}, fail: map[string][]error{}}
}

func (r *recorder) stage(name string, deps ...string) Stage {
	return Stage{
		Name: name,
		Deps: deps,
		Do: func(ctx context.Context) error {
			r.mu.Lock()
			r.calls[name]++
			r.active++
			if r.active > r.maxParallel {
				r.maxParallel = r.active
			}
			r.order = append(r.order, name)
			errs := r.fail[name]
			var err error
			if len(errs) > 0 {
				err = errs[0]
				if len(errs) > 1 {
					r.fail[name] = errs[1:]
				} else {
					delete(r.fail, name)
				}
			}
			r.mu.Unlock()
			// Yield so concurrent scheduling is exercised.
			time.Sleep(time.Millisecond)
			r.mu.Lock()
			r.active--
			r.mu.Unlock()
			return err
		},
	}
}

func (r *recorder) failOnce(name string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fail[name] = append(r.fail[name], err)
}

func (r *recorder) callCount(name string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls[name]
}

func (r *recorder) maxConc() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.maxParallel
}

func TestSerialHappyPath(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c")}

	if err := ex.Run(context.Background(), "op1", plan); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := ex.Watermark("op1"); got != 3 {
		t.Errorf("Watermark = %d; want 3", got)
	}
	if got := ex.Confirmed("op1"); len(got) != 3 || got[0] != "a" || got[2] != "c" {
		t.Errorf("Confirmed = %v; want [a b c] in order", got)
	}
	for _, name := range []string{"a", "b", "c"} {
		if rec.callCount(name) != 1 {
			t.Errorf("stage %q executed %d times; want 1", name, rec.callCount(name))
		}
		if !ex.IsConfirmed("op1", name) {
			t.Errorf("stage %q not confirmed", name)
		}
	}
}

func TestRetrySkipsConfirmedStages(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	boom := errors.New("stage b boom")
	rec.failOnce("b", boom)

	// "d" depends on "b"; it must be skipped on the first attempt while the
	// independent stage "c" still runs.
	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c"), rec.stage("d", "b")}

	err := ex.Run(context.Background(), "k", plan)
	if !errors.Is(err, boom) {
		t.Fatalf("first Run error = %v; want %v", err, boom)
	}
	if got := ex.Watermark("k"); got != 2 {
		t.Errorf("Watermark after partial failure = %d; want 2 (a, c)", got)
	}
	if rec.callCount("d") != 0 {
		t.Errorf("dependent stage d ran %d times; want 0 (its dependency failed)", rec.callCount("d"))
	}

	// Retry the same operation with the same idempotency key.
	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatalf("retry Run: %v", err)
	}
	if got := ex.Watermark("k"); got != 4 {
		t.Errorf("Watermark after retry = %d; want 4", got)
	}
	// a and c were confirmed on the first attempt and must not be repeated;
	// b ran twice (failed then succeeded); d once.
	wantCalls := map[string]int{"a": 1, "b": 2, "c": 1, "d": 1}
	for name, want := range wantCalls {
		if got := rec.callCount(name); got != want {
			t.Errorf("stage %q executed %d times across attempts; want %d", name, got, want)
		}
	}
}

func TestRetryWithStageSubset(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	boom := errors.New("b fail")
	rec.failOnce("b", boom)

	full := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c")}
	if err := ex.Run(context.Background(), "k", full); !errors.Is(err, boom) {
		t.Fatalf("first Run = %v; want %v", err, boom)
	}
	// A retry that only carries the unconfirmed work.
	if err := ex.Run(context.Background(), "k", []Stage{rec.stage("b")}); err != nil {
		t.Fatalf("subset retry: %v", err)
	}
	if got := ex.Watermark("k"); got != 3 {
		t.Errorf("Watermark = %d; want 3 retained across subsets", got)
	}
	if rec.callCount("a") != 1 || rec.callCount("c") != 1 {
		t.Errorf("confirmed stages re-run unexpectedly: a=%d c=%d", rec.callCount("a"), rec.callCount("c"))
	}
}

func TestOriginalErrorPreservedAcrossRetries(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	original := errors.New("original stage failure")
	rec.failOnce("b", original)

	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c")}
	if err := ex.Run(context.Background(), "k", plan); !errors.Is(err, original) {
		t.Fatalf("first Run = %v; want original", err)
	}

	// Second attempt is canceled before the unconfirmed stage can run. Its
	// returned error must still surface the original failure.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := ex.Run(ctx, "k", plan)
	if err == nil {
		t.Fatal("retry with canceled context returned nil; want error")
	}
	if !errors.Is(err, original) {
		t.Errorf("retry error %v lost the original error %v", err, original)
	}

	// Once a retry finishes the remaining work, the operation succeeds and
	// the stale error is cleared.
	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatalf("successful retry: %v", err)
	}
}

func TestDifferentKeysAreIndependent(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	plan := []Stage{rec.stage("a"), rec.stage("b")}
	for _, key := range []Key{"one", "two", "three"} {
		if err := ex.Run(context.Background(), key, plan); err != nil {
			t.Fatalf("Run(%q): %v", key, err)
		}
	}
	for _, name := range []string{"a", "b"} {
		if got := rec.callCount(name); got != 3 {
			t.Errorf("stage %q ran %d times; want once per distinct key (3)", name, got)
		}
	}
	if got := ex.Watermark("nonexistent"); got != 0 {
		t.Errorf("Watermark for unknown key = %d; want 0", got)
	}
}

func TestConcurrentSameKeyAttemptsDoNotDuplicate(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c")}

	var wg sync.WaitGroup
	errs := make([]error, 5)
	for i := range errs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = ex.Run(context.Background(), "same", plan)
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("attempt %d: %v", i, err)
		}
	}
	for _, name := range []string{"a", "b", "c"} {
		if got := rec.callCount(name); got != 1 {
			t.Errorf("stage %q ran %d times under concurrent duplicate attempts; want 1", name, got)
		}
	}
}

func TestConcurrentStages(t *testing.T) {
	ex := NewExecutor(WithConcurrent())
	rec := newRecorder()
	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c"), rec.stage("d", "b")}

	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := rec.maxConc(); got < 2 {
		t.Errorf("independent stages did not run in parallel, max concurrency = %d", got)
	}
	if got := ex.Watermark("k"); got != 4 {
		t.Errorf("Watermark = %d; want 4", got)
	}
}

func TestConcurrentFailureSkipsDependentsAndResumes(t *testing.T) {
	ex := NewExecutor(WithConcurrent())
	rec := newRecorder()
	boom := errors.New("b concurrent boom")
	rec.failOnce("b", boom)

	// d depends on b; a and c are independent and must still complete.
	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c"), rec.stage("d", "b")}
	if err := ex.Run(context.Background(), "k", plan); !errors.Is(err, boom) {
		t.Fatalf("first Run = %v; want %v", err, boom)
	}
	if rec.callCount("d") != 0 {
		t.Errorf("dependent d ran %d times; want 0", rec.callCount("d"))
	}
	if rec.callCount("a") != 1 || rec.callCount("c") != 1 {
		t.Errorf("independent stages did not run: a=%d c=%d", rec.callCount("a"), rec.callCount("c"))
	}
	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatalf("retry: %v", err)
	}
	if rec.callCount("d") != 1 {
		t.Errorf("d ran %d times after resume; want 1", rec.callCount("d"))
	}
}

func TestMaxParallel(t *testing.T) {
	ex := NewExecutor(WithConcurrent(), WithMaxParallel(2))
	rec := newRecorder()
	plan := []Stage{rec.stage("a"), rec.stage("b"), rec.stage("c"), rec.stage("d"), rec.stage("e")}
	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := rec.maxConc(); got > 2 {
		t.Errorf("max parallel stages = %d; want <= 2", got)
	}
}

func TestCanceledContextConfirmsNothing(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	plan := []Stage{rec.stage("a"), rec.stage("b")}
	err := ex.Run(ctx, "k", plan)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Run = %v; want context.Canceled", err)
	}
	if got := ex.Watermark("k"); got != 0 {
		t.Errorf("Watermark = %d; want 0 after pre-canceled run", got)
	}
	if rec.callCount("a") != 0 || rec.callCount("b") != 0 {
		t.Errorf("stages ran after cancellation: a=%d b=%d", rec.callCount("a"), rec.callCount("b"))
	}
}

func TestForget(t *testing.T) {
	ex := NewExecutor()
	rec := newRecorder()
	plan := []Stage{rec.stage("a")}
	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatal(err)
	}
	ex.Forget("k")
	if got := ex.Watermark("k"); got != 0 {
		t.Errorf("Watermark after Forget = %d; want 0", got)
	}
	if err := ex.Run(context.Background(), "k", plan); err != nil {
		t.Fatal(err)
	}
	if rec.callCount("a") != 2 {
		t.Errorf("stage ran %d times; want 2 after Forget restarted the operation", rec.callCount("a"))
	}
}

func TestEmptyPlan(t *testing.T) {
	ex := NewExecutor()
	if err := ex.Run(context.Background(), "k", nil); err != nil {
		t.Fatalf("empty plan: %v", err)
	}
	if got := ex.Keys(); len(got) != 0 {
		t.Errorf("empty plan recorded state %v", got)
	}
}

func TestValidation(t *testing.T) {
	ex := NewExecutor()
	ok := Stage{Name: "ok", Do: func(context.Context) error { return nil }}
	noDo := Stage{Name: "nodo"}
	noName := Stage{Do: func(context.Context) error { return nil }}
	cases := []struct {
		name   string
		stages []Stage
	}{
		{"empty name", []Stage{ok, noName}},
		{"nil do", []Stage{ok, noDo}},
		{"duplicate name", []Stage{ok, ok}},
		{"unknown dependency", []Stage{ok, {Name: "x", Deps: []string{"ghost"}, Do: func(context.Context) error { return nil }}}},
		{"later dependency", []Stage{{Name: "x", Do: func(context.Context) error { return nil }}, {Name: "y", Deps: []string{"z"}, Do: func(context.Context) error { return nil }}, {Name: "z", Do: func(context.Context) error { return nil }}}},
		{"self dependency", []Stage{ok, {Name: "self", Deps: []string{"self"}, Do: func(context.Context) error { return nil }}}},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			key := Key(fmt.Sprintf("validate-%d", i))
			if err := ex.Run(context.Background(), key, tc.stages); err == nil {
				t.Fatal("want validation error, got nil")
			}
		})
	}
}

func TestRetrySubsetWithConfirmedExternalDependency(t *testing.T) {
	for _, opt := range [][]Option{nil, {WithConcurrent()}} {
		ex := NewExecutor(opt...)
		rec := newRecorder()
		boom := errors.New("b fail")
		rec.failOnce("b", boom)

		full := []Stage{rec.stage("a"), rec.stage("b")}
		if err := ex.Run(context.Background(), "k", full); !errors.Is(err, boom) {
			t.Fatalf("first Run = %v; want %v", err, boom)
		}
		// Subset plan carrying a stage whose dependency is absent from the
		// subset but was confirmed by the earlier full plan.
		subset := []Stage{{
			Name: "b",
			Deps: []string{"a"},
			Do:   func(context.Context) error { return nil },
		}}
		if err := ex.Run(context.Background(), "k", subset); err != nil {
			t.Fatalf("subset retry after external dep confirmed: %v", err)
		}

		// Without the external dependency having been confirmed, the subset
		// must be rejected.
		bad := []Stage{{Name: "z", Deps: []string{"ghost"}, Do: func(context.Context) error { return nil }}}
		if err := ex.Run(context.Background(), "other", bad); err == nil {
			t.Fatal("want error for unconfirmed external dependency, got nil")
		}
	}
}

func TestStageErrorWrapping(t *testing.T) {
	ex := NewExecutor()
	sentinel := errors.New("sentinel")
	plan := []Stage{{
		Name: "only",
		Do:   func(context.Context) error { return fmt.Errorf("apply failed: %w", sentinel) },
	}}
	err := ex.Run(context.Background(), "k", plan)
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v does not wrap sentinel", err)
	}
}

func TestSerialExecutionIsDefault(t *testing.T) {
	// The default executor must not run stages in parallel: this is the
	// single-goroutine path that callers rely on for ordered side effects.
	ex := NewExecutor()
	rec := newRecorder()
	release := make(chan struct{})
	started := make(chan struct{}, 4)
	plan := []Stage{
		{Name: "a", Do: func(context.Context) error { started <- struct{}{}; <-release; return nil }},
		rec.stage("b"),
		rec.stage("c"),
	}
	done := make(chan error, 1)
	go func() { done <- ex.Run(context.Background(), "k", plan) }()
	<-started
	// While "a" is blocked, later stages must not have executed.
	time.Sleep(10 * time.Millisecond)
	if got := rec.callCount("b") + rec.callCount("c"); got != 0 {
		t.Errorf("stages ran concurrently on serial executor: calls=%d", got)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatalf("Run: %v", err)
	}
}
