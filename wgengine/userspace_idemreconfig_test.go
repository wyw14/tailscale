// Copyright (c) Tailscale Inc & contributors
// SPDX-License-Identifier: BSD-3-Clause

package wgengine

import (
	"errors"
	"fmt"
	"net/netip"
	"runtime"
	"sync"
	"testing"

	"tailscale.com/envknob"
	"tailscale.com/health"
	"tailscale.com/net/dns"
	"tailscale.com/net/netaddr"
	"tailscale.com/tailcfg"
	"tailscale.com/types/logger"
	"tailscale.com/types/netmap"
	"tailscale.com/util/eventbus/eventbustest"
	"tailscale.com/util/usermetric"
	"tailscale.com/wgengine/router"
	"tailscale.com/wgengine/wgcfg"
)

// flakyCountingRouter is a router.Router that records Set calls and fails the
// first failRemaining calls with err before succeeding.
type flakyCountingRouter struct {
	mu            sync.Mutex
	sets          int
	failRemaining int
	err           error
}

func (r *flakyCountingRouter) Up() error { return nil }
func (r *flakyCountingRouter) Close() error {
	return nil
}
func (r *flakyCountingRouter) Set(*router.Config) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sets++
	if r.failRemaining > 0 {
		r.failRemaining--
		return r.err
	}
	return nil
}

func (r *flakyCountingRouter) setCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sets
}

// countingOSConfigurator is a dns.OSConfigurator that records SetDNS calls.
type countingOSConfigurator struct {
	mu   sync.Mutex
	sets int
	err  error
}

func (c *countingOSConfigurator) SetDNS(dns.OSConfig) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sets++
	return c.err
}
func (c *countingOSConfigurator) SupportsSplitDNS() bool { return false }
func (c *countingOSConfigurator) Close() error           { return nil }
func (c *countingOSConfigurator) GetBaseConfig() (dns.OSConfig, error) {
	return dns.OSConfig{}, dns.ErrGetBaseConfigNotSupported
}
func (c *countingOSConfigurator) setCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sets
}

func newIdemTestEngine(t *testing.T) (*userspaceEngine, *flakyCountingRouter, *countingOSConfigurator) {
	t.Helper()
	bus := eventbustest.NewBus(t)
	ht := health.NewTracker(bus)
	reg := new(usermetric.Registry)
	e, err := NewFakeUserspaceEngine(logger.Discard, 0, ht, reg, bus)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(e.Close)
	ue := e.(*userspaceEngine)

	r := &flakyCountingRouter{}
	ue.router = r
	osCfg := &countingOSConfigurator{}
	ue.dns = dns.NewManager(logger.Discard, osCfg, ht, ue.dialer, nil, nil, runtime.GOOS, bus)
	return ue, r, osCfg
}

func idemReconfigConfigs() (*wgcfg.Config, *router.Config, *dns.Config) {
	cfg := &wgcfg.Config{
		Addresses: []netip.Prefix{netip.PrefixFrom(netaddr.IPv4(100, 100, 99, 1), 32)},
	}
	return cfg, &router.Config{}, &dns.Config{}
}

// TestIdempotentReconfigRetrySkipsConfirmedDNS verifies the opt-in
// idempotently-keyed reconfig path: when router configuration fails after DNS
// has already been configured, a retry of the same policy update re-runs only
// the unconfirmed router stage, leaves the confirmed DNS stage alone, and
// ultimately succeeds. The original failure is surfaced on the first attempt.
func TestIdempotentReconfigRetrySkipsConfirmedDNS(t *testing.T) {
	envknob.SetenvForTest(t, "TS_DEBUG_IDEMPOTENT_RECONFIG", "true")

	ue, r, osCfg := newIdemTestEngine(t)

	routerErr := fmt.Errorf("router idem boom")
	r.failRemaining = 1
	r.err = routerErr

	cfg, routerCfg, dnsCfg := idemReconfigConfigs()
	ue.SetSelfNode((&netmap.NetworkMap{
		Peers: nodeViews([]*tailcfg.Node{{ID: 1, Key: nkFromHex("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")}}),
	}).SelfNode)

	err := ue.Reconfig(cfg, routerCfg, dnsCfg)
	if !errors.Is(err, routerErr) {
		t.Fatalf("first Reconfig error = %v; want %v", err, routerErr)
	}
	if got := r.setCount(); got != 1 {
		t.Fatalf("router.Set called %d times on first attempt; want 1", got)
	}
	if got := osCfg.setCount(); got != 1 {
		t.Fatalf("SetDNS called %d times on first attempt; want 1 (DNS independent of router)", got)
	}

	// Retry the identical update: the router stage is unconfirmed and runs
	// again; the DNS stage is confirmed and must be skipped.
	if err := ue.Reconfig(cfg, routerCfg, dnsCfg); err != nil {
		t.Fatalf("retry Reconfig: %v", err)
	}
	if got := r.setCount(); got != 2 {
		t.Errorf("router.Set total calls = %d; want 2 (failed then retried)", got)
	}
	if got := osCfg.setCount(); got != 1 {
		t.Errorf("SetDNS total calls = %d; want 1 (confirmed stage must not be repeated)", got)
	}
}

// TestIdempotentReconfigNewGenerationAppliesAll ensures a different policy
// update gets a fresh idempotency key and applies all of its stages.
func TestIdempotentReconfigNewGenerationAppliesAll(t *testing.T) {
	envknob.SetenvForTest(t, "TS_DEBUG_IDEMPOTENT_RECONFIG", "true")

	ue, r, osCfg := newIdemTestEngine(t)
	ue.SetSelfNode((&netmap.NetworkMap{
		Peers: nodeViews([]*tailcfg.Node{{ID: 1, Key: nkFromHex("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")}}),
	}).SelfNode)

	cfg, routerCfg, dnsCfg := idemReconfigConfigs()
	if err := ue.Reconfig(cfg, routerCfg, dnsCfg); err != nil {
		t.Fatalf("first Reconfig: %v", err)
	}

	// A genuinely new router policy makes this a new operation; DNS is
	// historically re-evaluated alongside a router change.
	newRouterCfg := &router.Config{
		LocalAddrs: []netip.Prefix{netip.PrefixFrom(netaddr.IPv4(100, 100, 99, 1), 32)},
	}
	if err := ue.Reconfig(cfg, newRouterCfg, dnsCfg); err != nil {
		t.Fatalf("second Reconfig: %v", err)
	}
	if got := r.setCount(); got != 2 {
		t.Errorf("router.Set total calls = %d; want 2 across two generations", got)
	}
	if got := osCfg.setCount(); got != 2 {
		t.Errorf("SetDNS total calls = %d; want 2 across two generations", got)
	}
}

// TestIdempotentReconfigDisabledByDefault verifies that without the knob the
// new executor is never constructed and the historical path is used.
func TestIdempotentReconfigDisabledByDefault(t *testing.T) {
	ue, _, _ := newIdemTestEngine(t)
	ue.SetSelfNode((&netmap.NetworkMap{
		Peers: nodeViews([]*tailcfg.Node{{ID: 1, Key: nkFromHex("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")}}),
	}).SelfNode)

	cfg, routerCfg, dnsCfg := idemReconfigConfigs()
	if err := ue.Reconfig(cfg, routerCfg, dnsCfg); err != nil {
		t.Fatalf("Reconfig: %v", err)
	}
	if ue.idemReconfig != nil {
		t.Errorf("idempotent executor constructed with the knob off; default path must be used")
	}
	if ue.idemKey != "" {
		t.Errorf("idempotency key %q recorded with the knob off", ue.idemKey)
	}
}
