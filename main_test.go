package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestCanonicalPointID(t *testing.T) {
	cases := []struct {
		input    interface{}
		expected string
	}{
		{"0000e554-7b69-5e18-840b-abe326eb94f1", "0000e554-7b69-5e18-840b-abe326eb94f1"},
		{int64(42), "42"},
		{float64(105), "105"},
		{uint64(999), "999"},
	}

	for _, c := range cases {
		actual := canonicalPointID(c.input)
		if actual != c.expected {
			t.Errorf("canonicalPointID(%v) = %q, want %q", c.input, actual, c.expected)
		}
	}
}

func TestChooseProjectionDimension(t *testing.T) {
	// Frontpocket vectors have 3072 dimensions
	vec3072 := make([]float64, 3072)
	points := []QPoint{
		{ID: "point-1", Vector: vec3072},
		{ID: "point-2", Vector: vec3072},
	}

	dim := chooseProjectionDimension(points)
	if dim != 3072 {
		t.Errorf("chooseProjectionDimension = %d, want 3072", dim)
	}
}

func TestPickSearchVector(t *testing.T) {
	vec3072 := make([]float64, 3072)
	for i := range vec3072 {
		vec3072[i] = 0.01
	}

	resVec, name := pickSearchVector(vec3072, "")
	if name != "" {
		t.Errorf("pickSearchVector name = %q, want empty", name)
	}
	slice, ok := resVec.([]float64)
	if !ok || len(slice) != 3072 {
		t.Fatalf("pickSearchVector slice len = %v, want 3072", len(slice))
	}
}

func TestEmbedQueryWithOpenRouter(t *testing.T) {
	expectedVector := []float64{0.1, 0.2, 0.3}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.Header.Get("X-Title") != "VectorView" {
			http.Error(w, "missing X-Title", http.StatusBadRequest)
			return
		}
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if req["model"] != "google/gemini-embedding-2-preview" {
			http.Error(w, "unexpected model", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": expectedVector, "index": 0},
			},
		})
	}))
	defer server.Close()

	ctx := context.Background()
	emb, err := embedQueryWithOpenRouter(ctx, server.URL, "test-key", "google/gemini-embedding-2-preview", "test query", 3072)
	if err != nil {
		t.Fatalf("embedQueryWithOpenRouter failed: %v", err)
	}
	if len(emb) != len(expectedVector) {
		t.Errorf("got dim %d, want %d", len(emb), len(expectedVector))
	}
}

func TestEmbedQueryWithOpenAI(t *testing.T) {
	expectedVector := []float64{0.5, 0.6}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-openai-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"embedding": expectedVector, "index": 0},
			},
		})
	}))
	defer server.Close()

	ctx := context.Background()
	emb, err := embedQueryWithOpenAI(ctx, server.URL, "test-openai-key", "text-embedding-3-small", "test query", 1536)
	if err != nil {
		t.Fatalf("embedQueryWithOpenAI failed: %v", err)
	}
	if len(emb) != len(expectedVector) {
		t.Errorf("got dim %d, want %d", len(emb), len(expectedVector))
	}
}

func TestDetectCollectionEmbedding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/collections/frontpocket_memory":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"config": map[string]interface{}{
						"params": map[string]interface{}{
							"vectors": map[string]interface{}{
								"size": 3072,
							},
						},
					},
				},
			})
		case "/collections/frontpocket_memory/points/scroll":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"points": []map[string]interface{}{
						{
							"id": "test-id",
							"payload": map[string]interface{}{
								"embedding_provider":   "openrouter",
								"embedding_model":      "google/gemini-embedding-2",
								"embedding_dimensions": 3072,
							},
						},
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	q := newQdrant(server.URL, "")
	ctx := context.Background()
	provider, model, dims := q.detectCollectionEmbedding(ctx, "frontpocket_memory")
	if provider != "openrouter" {
		t.Errorf("detectCollectionEmbedding provider = %q, want openrouter", provider)
	}
	if model != "google/gemini-embedding-2" {
		t.Errorf("detectCollectionEmbedding model = %q, want google/gemini-embedding-2", model)
	}
	if dims != 3072 {
		t.Errorf("detectCollectionEmbedding dims = %d, want 3072", dims)
	}
}

func TestWriteJSONErrorEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSONError(rec, http.StatusBadRequest, errCodeBadRequest, "bad input")

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	var body apiErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode error envelope: %v (body=%q)", err, rec.Body.String())
	}
	if body.Error.Code != errCodeBadRequest {
		t.Errorf("code = %q, want %q", body.Error.Code, errCodeBadRequest)
	}
	if body.Error.Message != "bad input" {
		t.Errorf("message = %q, want bad input", body.Error.Message)
	}
}

func TestWriteProjectionErrorMapsSaturation(t *testing.T) {
	rec := httptest.NewRecorder()
	writeProjectionError(rec, errProjectionBusy)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusTooManyRequests)
	}
	var body apiErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode envelope: %v", err)
	}
	if body.Error.Code != errCodeSaturated {
		t.Errorf("code = %q, want %q", body.Error.Code, errCodeSaturated)
	}
}

func TestAuthMiddleware(t *testing.T) {
	okHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok")
	})

	cases := []struct {
		name       string
		token      string
		method     string
		path       string
		target     string
		header     map[string]string
		wantStatus int
	}{
		{
			name:       "disabled without configured token",
			token:      "",
			path:       "/api/points",
			target:     "/api/points",
			wantStatus: http.StatusOK,
		},
		{
			name:       "missing credentials",
			token:      "secret",
			path:       "/api/points",
			target:     "/api/points",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "bearer token",
			token:      "secret",
			path:       "/api/points",
			target:     "/api/points",
			header:     map[string]string{"Authorization": "Bearer secret"},
			wantStatus: http.StatusOK,
		},
		{
			name:       "custom header",
			token:      "secret",
			path:       "/api/points",
			target:     "/api/points",
			header:     map[string]string{"X-VectorView-Token": "secret"},
			wantStatus: http.StatusOK,
		},
		{
			name:       "query token",
			token:      "secret",
			path:       "/api/stream",
			target:     "/api/stream?token=secret",
			wantStatus: http.StatusOK,
		},
		{
			name:       "wrong token",
			token:      "secret",
			path:       "/api/points",
			target:     "/api/points",
			header:     map[string]string{"Authorization": "Bearer nope"},
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "static assets bypass auth",
			token:      "secret",
			path:       "/index.html",
			target:     "/index.html",
			wantStatus: http.StatusOK,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			handler := newAuthMiddleware(c.token, okHandler)
			target := c.target
			if target == "" {
				target = c.path
			}
			req := httptest.NewRequest(http.MethodGet, target, nil)
			for key, value := range c.header {
				req.Header.Set(key, value)
			}
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != c.wantStatus {
				t.Fatalf("status = %d, want %d (body=%q)", rec.Code, c.wantStatus, rec.Body.String())
			}
			if c.wantStatus == http.StatusUnauthorized {
				var body apiErrorResponse
				if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
					t.Fatalf("decode unauthorized body: %v", err)
				}
				if body.Error.Code != errCodeUnauthorized {
					t.Errorf("code = %q, want %q", body.Error.Code, errCodeUnauthorized)
				}
			}
		})
	}
}

func TestDiskProjectionCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cache, err := newDiskProjectionCache(dir, time.Minute, 1<<20)
	if err != nil {
		t.Fatalf("newDiskProjectionCache: %v", err)
	}
	if cache == nil {
		t.Fatal("expected cache, got nil")
	}

	key := projectionCacheKey("meta_bridge", 10, "pca", "")
	value := &PointsResponse{
		Points: []PointData{{ID: "point-1", X: 1, Y: 2, Z: 3}},
		Total:  1,
	}
	if err := cache.Set(context.Background(), key, value); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, ok, err := cache.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !ok || got == nil {
		t.Fatal("expected cache hit")
	}
	if !got.CachedResult {
		t.Error("expected CachedResult=true on cache hit")
	}
	if len(got.Points) != 1 || got.Points[0].ID != "point-1" {
		t.Errorf("unexpected points: %+v", got.Points)
	}

	if _, ok, _ := cache.Get(context.Background(), projectionCacheKey("meta_bridge", 11, "pca", "")); ok {
		t.Error("expected miss for unknown key")
	}
}

func TestDiskProjectionCacheExpiry(t *testing.T) {
	dir := t.TempDir()
	cache, err := newDiskProjectionCache(dir, 20*time.Millisecond, 1<<20)
	if err != nil {
		t.Fatalf("newDiskProjectionCache: %v", err)
	}
	key := projectionCacheKey("meta_bridge", 10, "pca", "")
	if err := cache.Set(context.Background(), key, &PointsResponse{Total: 1, Points: []PointData{{ID: "a"}}}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	time.Sleep(60 * time.Millisecond)

	if _, ok, err := cache.Get(context.Background(), key); err != nil {
		t.Fatalf("Get: %v", err)
	} else if ok {
		t.Fatal("expected expired entry to miss")
	}
	if _, err := os.Stat(filepath.Join(dir, filepath.Base(cache.path(key)))); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected expired cache file to be removed, stat err = %v", err)
	}
}

func TestDiskProjectionCacheIgnoresCorruptEntry(t *testing.T) {
	dir := t.TempDir()
	cache, err := newDiskProjectionCache(dir, time.Minute, 1<<20)
	if err != nil {
		t.Fatalf("newDiskProjectionCache: %v", err)
	}
	key := projectionCacheKey("meta_bridge", 10, "pca", "")
	if err := os.WriteFile(cache.path(key), []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write corrupt entry: %v", err)
	}
	value, ok, err := cache.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ok || value != nil {
		t.Fatal("expected corrupt entry to be treated as a miss")
	}
}

func TestTieredProjectionCacheReadsSecondBackend(t *testing.T) {
	primary, err := newDiskProjectionCache(t.TempDir(), time.Minute, 1<<20)
	if err != nil {
		t.Fatalf("primary cache: %v", err)
	}
	secondary, err := newDiskProjectionCache(t.TempDir(), time.Minute, 1<<20)
	if err != nil {
		t.Fatalf("secondary cache: %v", err)
	}
	chain := &tieredProjectionCache{backends: []projectionCache{primary, secondary}}

	key := projectionCacheKey("meta_bridge", 10, "pca", "")
	if err := secondary.Set(context.Background(), key, &PointsResponse{Total: 3}); err != nil {
		t.Fatalf("secondary Set: %v", err)
	}

	got, ok, err := chain.Get(context.Background(), key)
	if err != nil {
		t.Fatalf("chain Get: %v", err)
	}
	if !ok || got == nil || got.Total != 3 {
		t.Fatalf("expected hit from secondary backend, got ok=%v value=%+v", ok, got)
	}

	if err := chain.Set(context.Background(), key, &PointsResponse{Total: 9}); err != nil {
		t.Fatalf("chain Set: %v", err)
	}
	if _, ok, _ := primary.Get(context.Background(), key); !ok {
		t.Error("expected chain Set to populate primary backend")
	}
}

func TestProjectionGuardLimitsConcurrency(t *testing.T) {
	guard := newProjectionGuard(1, 40*time.Millisecond)
	if guard.capacity() != 1 {
		t.Fatalf("capacity = %d, want 1", guard.capacity())
	}
	if err := guard.acquire(context.Background()); err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if guard.inUse() != 1 {
		t.Errorf("inUse = %d, want 1", guard.inUse())
	}

	start := time.Now()
	if err := guard.acquire(context.Background()); !errors.Is(err, errProjectionBusy) {
		t.Fatalf("second acquire err = %v, want errProjectionBusy", err)
	}
	if elapsed := time.Since(start); elapsed < 40*time.Millisecond {
		t.Errorf("second acquire returned after %v, expected to wait for the queue timeout", elapsed)
	}
	if guard.inUse() != 1 {
		t.Errorf("inUse after failed acquire = %d, want 1", guard.inUse())
	}

	guard.release()
	if guard.inUse() != 0 {
		t.Errorf("inUse after release = %d, want 0", guard.inUse())
	}
	if err := guard.acquire(context.Background()); err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	guard.release()
}

func TestProjectionGuardHonorsContextCancellation(t *testing.T) {
	guard := newProjectionGuard(1, 5*time.Second)
	if err := guard.acquire(context.Background()); err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	defer guard.release()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	if err := guard.acquire(ctx); !errors.Is(err, errProjectionBusy) {
		t.Fatalf("acquire err = %v, want errProjectionBusy", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("cancelled acquire took %v, expected an immediate return", elapsed)
	}
}

func TestProjectionFlightsDedupesConcurrentCalls(t *testing.T) {
	flights := newProjectionFlights()
	var calls int32
	release := make(chan struct{})

	fn := func() (PointsResponse, error) {
		atomic.AddInt32(&calls, 1)
		<-release
		return PointsResponse{Total: 7}, nil
	}

	const workers = 4
	results := make([]PointsResponse, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx], errs[idx] = flights.do("shared-key", fn)
		}(i)
	}

	time.Sleep(100 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("projection ran %d times, want 1", got)
	}
	for i := range results {
		if errs[i] != nil {
			t.Errorf("worker %d err = %v", i, errs[i])
		}
		if results[i].Total != 7 {
			t.Errorf("worker %d total = %d, want 7", i, results[i].Total)
		}
	}

	// A later call must not be served by the completed flight.
	if _, err := flights.do("shared-key", func() (PointsResponse, error) {
		return PointsResponse{Total: 1}, nil
	}); err != nil {
		t.Fatalf("sequential call err = %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("sequential call unexpectedly reused flight, calls = %d", got)
	}
}

func TestBuildHealthReportsQdrantState(t *testing.T) {
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"result": map[string]interface{}{
				"collections": []map[string]string{{"name": "meta_bridge"}},
			},
		})
	}))
	defer healthy.Close()

	guard := newProjectionGuard(3, time.Second)
	out := buildHealth(context.Background(), newQdrant(healthy.URL, ""), []string{"disk:/tmp"}, guard, time.Now())
	if out.Status != "ok" {
		t.Errorf("status = %q, want ok", out.Status)
	}
	if !out.Qdrant.Reachable {
		t.Errorf("expected qdrant reachable, error = %q", out.Qdrant.Error)
	}
	if out.Projections.Capacity != 3 {
		t.Errorf("capacity = %d, want 3", out.Projections.Capacity)
	}
	if len(out.Caches) != 1 {
		t.Errorf("caches = %v, want one entry", out.Caches)
	}

	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer down.Close()

	degraded := buildHealth(context.Background(), newQdrant(down.URL, ""), nil, guard, time.Now())
	if degraded.Status != "degraded" {
		t.Errorf("status = %q, want degraded", degraded.Status)
	}
	if degraded.Qdrant.Reachable {
		t.Error("expected qdrant to be unreachable")
	}
	if degraded.Caches == nil {
		t.Error("expected caches to marshal as an empty array, got nil")
	}
}
