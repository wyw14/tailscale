// Copyright (c) Tailscale Inc & contributors
// SPDX-License-Identifier: BSD-3-Clause

// Package stagedop runs multi-stage, side-effecting operations idempotently.
//
// Network jitter, context cancellation, or a caller retrying a request can
// cause the same logical operation (for example applying a policy update to a
// node's control-plane state and network components) to be attempted more
// than once. Naively re-running such an operation repeats side effects that
// already happened on a prior attempt.
//
// Two mechanisms prevent duplicate side effects:
//
//   - An idempotency [Key] identifies a logical operation. All [Executor.Run]
//     calls with the same key, including concurrent ones, are attempts of the
//     same operation and share its state.
//   - A stage is confirmed only after it completes successfully; the executor
//     records a per-key watermark of confirmed stages. A retry with the same
//     key executes only the stages the watermark has not confirmed yet.
//
// Stage dependencies may be expressed with [Stage.Deps]: a stage runs only
// once all of its dependencies are confirmed, and is skipped (left
// unconfirmed) if a dependency failed.
//
// The first error an operation produces is preserved across retries: if a
// later retry fails too, the returned error still wraps the original stage
// error, so callers can inspect it with [errors.Is] and [errors.As]. When a
// retry confirms every remaining stage, the operation is complete and the
// error is cleared.
//
// By default stages are applied one at a time, in plan order (the single
// goroutine path). [WithConcurrent] enables parallel execution of independent
// stages; it is opt-in because stage functions must be safe to call
// concurrently in that mode.
package stagedop

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"

	"tailscale.com/types/logger"
)

// Key is an idempotency key for a logical operation.
//
// All Run calls that share a key are attempts of the same operation: they
// share the operation's watermark, and their side-effecting stages are
// de-duplicated. Calls with different keys are independent operations.
type Key string

// Stage is one side-effecting step of an [operation].
type Stage struct {
	// Name identifies the stage within an operation. It must be non-empty and
	// unique within the plan. The watermark records confirmed stages by name.
	Name string

	// Deps optionally names stages (from the same plan) that must be
	// confirmed before this stage runs. If a dependency fails or is itself
	// skipped, this stage is not run and remains unconfirmed so that a later
	// retry can run it once the dependency succeeds. Deps must refer to
	// stages declared earlier in the plan.
	Deps []string

	// Do performs the stage's side effect.
	//
	// Do must be idempotent: it may be invoked more than once if a prior
	// attempt failed or was interrupted after the side effect happened but
	// before it could be confirmed. Do should return nil only once its side
	// effect has durably completed; that nil result is what confirms the
	// stage and advances the watermark.
	Do func(ctx context.Context) error
}

// Option configures an [Executor]. The zero-value options select the default
// behavior: serial stage execution and a discard logger.
type Option func(*Executor)

// WithConcurrent enables parallel execution of independent stages. It is
// disabled by default; when disabled, stages run one at a time in plan order
// on the calling goroutine.
//
// With it enabled, [Stage.Do] functions may run on different goroutines and
// must be safe for concurrent use whenever the stages are independent.
func WithConcurrent() Option {
	return func(e *Executor) { e.concurrent = true }
}

// WithMaxParallel bounds the number of stages that run simultaneously when
// [WithConcurrent] is enabled. A value <= 0 means no limit. It has no effect
// on the default serial execution path.
func WithMaxParallel(n int) Option {
	return func(e *Executor) { e.maxParallel = n }
}

// WithLogger installs a log function used for verbose progress messages.
func WithLogger(logf logger.Logf) Option {
	return func(e *Executor) { e.logf = logf }
}

// Executor runs idempotent, multi-stage operations.
//
// An Executor is safe for concurrent use by multiple goroutines; operations
// with different keys proceed independently, while attempts of the same key
// are serialized so a stage is never executed twice concurrently.
//
// The zero value is not ready for use; create one with [NewExecutor].
type Executor struct {
	concurrent  bool
	maxParallel int
	logf        logger.Logf

	mu  sync.Mutex
	ops map[Key]*opState
}

// NewExecutor returns an [Executor] configured with the given options.
func NewExecutor(opts ...Option) *Executor {
	e := &Executor{
		logf: logger.Discard,
		ops:  make(map[Key]*opState),
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// opState is the watermark and error bookkeeping for one logical operation.
type opState struct {
	// attemptMu serializes whole attempts of the same operation, so duplicate
	// (including concurrent) retries can never run the same stage at once.
	attemptMu sync.Mutex

	// mu guards confirmed, confirmedOrder and firstErr. Stages can complete
	// concurrently under [WithConcurrent].
	mu             sync.Mutex
	confirmed      map[string]bool
	confirmedOrder []string
	// firstErr is the original error this operation produced. It is kept
	// across attempts until the operation completes successfully.
	firstErr error
}

// Run executes the unconfirmed stages of the operation identified by key.
//
// Stages that a previous attempt with the same key already confirmed are not
// re-run. Stages are executed serially in plan order (the default) or in
// parallel when the executor was created with [WithConcurrent]. A stage whose
// dependency fails is skipped and stays unconfirmed, so a later retry picks
// it up after the dependency succeeds.
//
// The plan must be consistent across retries of the same key: stage names
// identify stages, so passing the same ordered plan (or a subset of it) on
// every attempt lets the watermark resume correctly.
//
// Run returns nil once every provided stage is confirmed. On failure it
// returns the stage error(s); when an earlier attempt already failed, the
// original error is joined in first and remains discoverable via
// [errors.Is]/[errors.As].
func (e *Executor) Run(ctx context.Context, key Key, stages []Stage) error {
	if len(stages) == 0 {
		return nil
	}
	if err := validate(stages); err != nil {
		return err
	}

	e.mu.Lock()
	op := e.ops[key]
	if op == nil {
		op = &opState{confirmed: make(map[string]bool)}
		e.ops[key] = op
	}
	e.mu.Unlock()

	// A dependency that is not part of this (possibly subset) plan is only
	// runnable if an earlier attempt of the same operation already confirmed
	// it.
	inPlan := make(map[string]bool, len(stages))
	for _, s := range stages {
		inPlan[s.Name] = true
	}
	for _, s := range stages {
		for _, dep := range s.Deps {
			if !inPlan[dep] && !op.isConfirmed(dep) {
				return fmt.Errorf("stagedop: stage %q depends on %q, which is neither in the plan nor confirmed", s.Name, dep)
			}
		}
	}

	op.attemptMu.Lock()
	defer op.attemptMu.Unlock()

	var errs []error
	if e.concurrent {
		errs = e.runConcurrent(ctx, op, stages)
	} else {
		errs = e.runSerial(ctx, op, stages)
	}

	attemptErr := errors.Join(errs...)
	allConfirmed := true
	for _, s := range stages {
		if !op.isConfirmed(s.Name) {
			allConfirmed = false
			break
		}
	}

	op.mu.Lock()
	if allConfirmed {
		// The operation (at least the provided plan) is complete. Drop any
		// error recorded by an earlier, now-superseded attempt.
		op.firstErr = nil
	} else if attemptErr != nil && op.firstErr == nil {
		op.firstErr = errs[0]
	}
	original := op.firstErr
	op.mu.Unlock()

	if allConfirmed {
		return nil
	}
	return joinErrors(original, attemptErr)
}

// runSerial applies stages in plan order on the calling goroutine. Stages
// independent of a failed stage still run; only the failed stage's
// dependents are skipped.
func (e *Executor) runSerial(ctx context.Context, op *opState, stages []Stage) []error {
	var errs []error
	for i := range stages {
		s := &stages[i]
		if op.isConfirmed(s.Name) {
			e.logf("[v1] stagedop: stage %q already confirmed; skipping", s.Name)
			continue
		}
		if !op.depsConfirmed(s.Deps) {
			// A dependency has not been confirmed by this attempt; leave this
			// stage unconfirmed for a future retry to pick up.
			e.logf("[v1] stagedop: stage %q waiting on unconfirmed dependency", s.Name)
			continue
		}
		if err := ctx.Err(); err != nil {
			errs = append(errs, err)
			break
		}
		if err := e.runStage(ctx, op, s); err != nil {
			errs = append(errs, err)
		}
	}
	return errs
}

// stageStatus values track a single concurrent attempt's stages.
const (
	stPending    = 0 // not yet started
	stRunning    = 1 // executing
	stConfirmed  = 2 // succeeded and confirmed
	stFailed     = 3 // returned an error
	stUnrunnable = 4 // skipped because a dependency failed
)

// runConcurrent applies independent stages in parallel. Stages in flight
// when another stage fails run to completion; everything else is left
// unconfirmed for a retry.
func (e *Executor) runConcurrent(ctx context.Context, op *opState, stages []Stage) []error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, e.maxParallel) // unbuffered-ish when maxParallel<=0

	mu := sync.Mutex{}
	cond := sync.NewCond(&mu)
	status := make([]int, len(stages))
	var errs []error
	var running int

	byName := make(map[string]int, len(stages))
	for i, s := range stages {
		byName[s.Name] = i
	}

	var wg sync.WaitGroup
	startStage := func(i int) {
		status[i] = stRunning
		running++
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := e.runStage(ctx, op, &stages[i])
			// Release the parallelism slot before waking the dispatcher, so a
			// waiting stage can acquire it when it handles the broadcast.
			if e.maxParallel > 0 {
				<-sem
			}
			mu.Lock()
			running--
			if err != nil {
				status[i] = stFailed
				errs = append(errs, err)
				cancel() // stop dispatching further stages
			} else {
				status[i] = stConfirmed
			}
			cond.Broadcast()
			mu.Unlock()
		}()
	}

	mu.Lock()
	for {
		progress := false
		for i := range stages {
			if status[i] != stPending {
				continue
			}
			if op.isConfirmed(stages[i].Name) {
				// Confirmed by an earlier attempt of the same operation.
				status[i] = stConfirmed
				progress = true
				continue
			}
			depState := stConfirmed
			for _, dep := range stages[i].Deps {
				// A dependency confirmed by an earlier attempt (possibly one
				// that isn't part of this subset plan) is satisfied.
				if op.isConfirmed(dep) {
					continue
				}
				switch st := status[byName[dep]]; st {
				case stFailed, stUnrunnable:
					depState = stUnrunnable
				case stPending, stRunning:
					if depState != stUnrunnable {
						depState = stPending
					}
				}
			}
			switch depState {
			case stConfirmed:
				if ctx.Err() != nil {
					continue
				}
				if e.maxParallel > 0 {
					select {
					case sem <- struct{}{}:
					default:
						continue // at capacity; revisit when a slot frees
					}
				}
				progress = true
				startStage(i)
			case stUnrunnable:
				status[i] = stUnrunnable
				progress = true
				e.logf("[v1] stagedop: stage %q waiting on failed dependency", stages[i].Name)
			}
		}
		pending := false
		for _, st := range status {
			if st == stPending || st == stRunning {
				pending = true
				break
			}
		}
		if !pending {
			break
		}
		if !progress && running == 0 {
			// Nothing is running and nothing could be started (typically a
			// canceled context); avoid deadlocking and stop.
			break
		}
		cond.Wait()
	}
	mu.Unlock()
	wg.Wait()
	for _, st := range status {
		if st == stPending {
			// Some stages never started (e.g. the context was canceled
			// before dispatch, or a dependency failed). A dependency failure
			// is already reported by the failed stage; only surface the
			// context error when there is no other cause.
			if len(errs) == 0 && ctx.Err() != nil {
				errs = append(errs, ctx.Err())
			}
			break
		}
	}
	return errs
}

func (e *Executor) runStage(ctx context.Context, op *opState, s *Stage) error {
	err := s.Do(ctx)
	if err == nil {
		op.confirm(s.Name)
		e.logf("[v1] stagedop: stage %q confirmed", s.Name)
		return nil
	}
	return fmt.Errorf("stage %q: %w", s.Name, err)
}

func (op *opState) confirm(name string) {
	op.mu.Lock()
	defer op.mu.Unlock()
	if !op.confirmed[name] {
		op.confirmed[name] = true
		op.confirmedOrder = append(op.confirmedOrder, name)
	}
}

func (op *opState) isConfirmed(name string) bool {
	op.mu.Lock()
	defer op.mu.Unlock()
	return op.confirmed[name]
}

func (op *opState) depsConfirmed(deps []string) bool {
	op.mu.Lock()
	defer op.mu.Unlock()
	for _, d := range deps {
		if !op.confirmed[d] {
			return false
		}
	}
	return true
}

// Watermark returns the number of stages confirmed so far for key. It is the
// operation's stage watermark: it starts at zero and only advances as stages
// are confirmed.
func (e *Executor) Watermark(key Key) int {
	op := e.state(key)
	if op == nil {
		return 0
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	return len(op.confirmed)
}

// Confirmed returns the names of the stages confirmed for key, in the order
// in which they were confirmed.
func (e *Executor) Confirmed(key Key) []string {
	op := e.state(key)
	if op == nil {
		return nil
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	return append([]string(nil), op.confirmedOrder...)
}

// IsConfirmed reports whether the named stage has been confirmed for key.
func (e *Executor) IsConfirmed(key Key, stage string) bool {
	op := e.state(key)
	if op == nil {
		return false
	}
	return op.isConfirmed(stage)
}

// Forget drops the recorded watermark (and any error) for key. Future Run
// calls with that key start the operation over. It is typically used to bound
// the executor's memory once an operation is superseded by a newer one.
func (e *Executor) Forget(key Key) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.ops, key)
}

func (e *Executor) state(key Key) *opState {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ops[key]
}

// Keys returns the idempotency keys the executor currently holds a watermark
// for, sorted. It is primarily intended for tests and diagnostics.
func (e *Executor) Keys() []Key {
	e.mu.Lock()
	defer e.mu.Unlock()
	keys := make([]Key, 0, len(e.ops))
	for k := range e.ops {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

// joinErrors returns the original error joined with the latest attempt's
// error, deduplicating the case where they are the same value so the original
// is never reported twice.
func joinErrors(original, attempt error) error {
	if original == nil {
		return attempt
	}
	if attempt == nil || original == attempt {
		return original
	}
	return errors.Join(original, attempt)
}

// validate checks a plan for structural errors that watermark resumption
// could not recover from. Dependencies must point to an earlier stage in the
// same plan, or to a stage omitted from this (subset) plan that a prior
// attempt confirmed; the latter is checked in [Executor.Run].
func validate(stages []Stage) error {
	index := make(map[string]int, len(stages))
	for i, s := range stages {
		if s.Name == "" {
			return fmt.Errorf("stagedop: stage at index %d has no name", i)
		}
		if s.Do == nil {
			return fmt.Errorf("stagedop: stage %q has no Do function", s.Name)
		}
		if _, dup := index[s.Name]; dup {
			return fmt.Errorf("stagedop: duplicate stage name %q", s.Name)
		}
		index[s.Name] = i
	}
	for i, s := range stages {
		for _, dep := range s.Deps {
			if dep == s.Name {
				return fmt.Errorf("stagedop: stage %q depends on itself", s.Name)
			}
			if j, ok := index[dep]; ok && j >= i {
				return fmt.Errorf("stagedop: stage %q depends on a later stage %q", s.Name, dep)
			}
		}
	}
	return nil
}
