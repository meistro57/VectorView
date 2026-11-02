package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type fakeQdrant struct {
	server    *httptest.Server
	points    []QPoint
	searchHit []searchPointHit
	failPath  string
}

func newFakeQdrant(t *testing.T) *fakeQdrant {
	t.Helper()
	fake := &fakeQdrant{
		points: []QPoint{
			{
				ID:      "p1",
				Vector:  []float64{0.1, 0.2, 0.3, 0.4},
				Payload: map[string]interface{}{"text": "alpha document", "file_source": "alpha.md"},
			},
			{
				ID:      "p2",
				Vector:  []float64{0.4, 0.3, 0.2, 0.1},
				Payload: map[string]interface{}{"text": "beta document", "file_source": "beta.md"},
			},
		},
		searchHit: []searchPointHit{
			{ID: "p1", Score: 0.99, Payload: map[string]interface{}{"text": "alpha document"}},
			{ID: "p2", Score: 0.42, Payload: map[string]interface{}{"text": "beta document"}},
		},
	}
	fake.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fake.failPath != "" && strings.HasPrefix(r.URL.Path, fake.failPath) {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/collections":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"collections": []map[string]string{{"name": "meta_bridge"}},
				},
			})
		case r.URL.Path == "/collections/meta_bridge":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"points_count": len(fake.points),
					"config": map[string]interface{}{
						"params": map[string]interface{}{
							"vectors": map[string]interface{}{"size": 4, "distance": "Cosine"},
						},
					},
				},
			})
		case strings.HasSuffix(r.URL.Path, "/points/scroll"):
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{"points": fake.points, "next_page_offset": nil},
			})
		case strings.HasSuffix(r.URL.Path, "/points/search"):
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": fake.searchHit})
		case strings.HasSuffix(r.URL.Path, "/points"):
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": fake.points})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(fake.server.Close)
	return fake
}

func newTestServer(t *testing.T, fake *fakeQdrant) (*server, *http.ServeMux) {
	t.Helper()
	app := &server{
		q:                        newQdrant(fake.server.URL, ""),
		cacheLabels:              []string{},
		projections:              newProjectionGuard(2, time.Second),
		flights:                  newProjectionFlights(),
		startedAt:                time.Now(),
		defaultCollection:        "meta_bridge",
		maxPoints:                100,
		qdrantURL:                fake.server.URL,
		inlinePCAFallbackMax:     100,
		python:                   pythonBin(),
		hub:                      newStreamHub(nil),
		streamHeartbeatSeconds:   5,
		streamMaxEventsPerSecond: 30,
		wsPingSeconds:            5,
	}
	return app, app.routes()
}

func TestHandleHealth(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var body healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode health: %v", err)
	}
	if body.Status != "ok" || !body.Qdrant.Reachable {
		t.Errorf("health = %+v, want ok + reachable", body)
	}
}

func TestHandleCollections(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/collections", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var metas []CollectionMeta
	if err := json.Unmarshal(rec.Body.Bytes(), &metas); err != nil {
		t.Fatalf("decode collections: %v", err)
	}
	if len(metas) != 1 || metas[0].Name != "meta_bridge" {
		t.Fatalf("collections = %+v, want one meta_bridge entry", metas)
	}
	if metas[0].VectorSize != 4 || !metas[0].ProjectionReady {
		t.Errorf("collection meta = %+v, want vector size 4 and ready", metas[0])
	}
}

func TestHandleCollectionsReportsQdrantFailure(t *testing.T) {
	fake := newFakeQdrant(t)
	fake.failPath = "/collections"
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/collections", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	var body apiErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode error envelope: %v", err)
	}
	if body.Error.Code != errCodeQdrant {
		t.Errorf("code = %q, want %q", body.Error.Code, errCodeQdrant)
	}
	if strings.Contains(rec.Body.String(), "boom") {
		t.Error("internal qdrant detail leaked to the client")
	}
}

func TestHandlePointsServesCacheAndMarksCached(t *testing.T) {
	fake := newFakeQdrant(t)
	app, mux := newTestServer(t, fake)

	cache, err := newDiskProjectionCache(t.TempDir(), time.Minute, 1<<20)
	if err != nil {
		t.Fatalf("disk cache: %v", err)
	}
	app.pointsCache = cache
	cached := &PointsResponse{
		Points: []PointData{{ID: "p1", X: 1, Y: 2, Z: 3}},
		Total:  1,
	}
	if err := cache.Set(context.Background(), projectionCacheKey("meta_bridge", 100, "pca", ""), cached); err != nil {
		t.Fatalf("prime cache: %v", err)
	}

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/points?collection=meta_bridge&limit=100", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var body PointsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode points: %v", err)
	}
	if !body.CachedResult {
		t.Error("expected cached_result=true")
	}
	if body.Total != 1 {
		t.Errorf("total = %d, want 1", body.Total)
	}
}

func TestHandlePointsRejectsBadParams(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	cases := []struct {
		target string
		code   string
	}{
		{"/api/points?collection=meta_bridge&append_from=nope", errCodeBadRequest},
		{"/api/points?collection=meta_bridge&append_from=50&limit=10", errCodeBadRequest},
		{"/api/points?collection=meta_bridge&projection=banana", errCodeBadRequest},
	}

	for _, c := range cases {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, c.target, nil))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s status = %d, want 400", c.target, rec.Code)
		}
		var body apiErrorResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("%s decode envelope: %v", c.target, err)
		}
		if body.Error.Code != c.code {
			t.Errorf("%s code = %q, want %q", c.target, body.Error.Code, c.code)
		}
	}
}

func TestHandleSearchRequiresQueryAndProjects(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/search?collection=meta_bridge&q=alpha&limit=2", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var body PointsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode search: %v", err)
	}
	if len(body.Points) != 2 {
		t.Fatalf("points = %d, want 2", len(body.Points))
	}
	if body.Projection == nil || body.Projection.Method == "" {
		t.Error("expected projection metadata on search response")
	}
}

func TestHandleSemanticSearchRequiresQuery(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/semantic-search?collection=meta_bridge", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	var body apiErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode envelope: %v", err)
	}
	if body.Error.Code != errCodeBadRequest {
		t.Errorf("code = %q, want %q", body.Error.Code, errCodeBadRequest)
	}
}

func TestHandleHighlightPublishesEvent(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	payload := `{"collection":"meta_bridge","ids":["p2","p1"],"focus_id":"p1"}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/highlight", strings.NewReader(payload))
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var event highlightEvent
	if err := json.Unmarshal(rec.Body.Bytes(), &event); err != nil {
		t.Fatalf("decode highlight: %v", err)
	}
	if len(event.IDs) != 2 || event.EventID == "" {
		t.Errorf("event = %+v, want two ids and an event id", event)
	}

	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/highlight?collection=meta_bridge&since="+event.EventID, nil))
	if rec.Code != http.StatusNoContent {
		t.Errorf("poll with matching since = %d, want 204", rec.Code)
	}
}

func TestHandleSimilarReturnsNeighbors(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/collections/meta_bridge/points/p1/similar?limit=5", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var body similarResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode similar: %v", err)
	}
	if len(body.Neighbors) != 1 || canonicalPointID(body.Neighbors[0].ID) != "p2" {
		t.Errorf("neighbors = %+v, want only p2", body.Neighbors)
	}
}

func TestHandleSimilarNotFound(t *testing.T) {
	fake := newFakeQdrant(t)
	fake.points = nil
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/collections/meta_bridge/points/missing/similar", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (body=%q)", rec.Code, rec.Body.String())
	}
	var body apiErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode envelope: %v", err)
	}
	if body.Error.Code != errCodeNotFound {
		t.Errorf("code = %q, want %q", body.Error.Code, errCodeNotFound)
	}
}

func TestHandlePointsRejectsWrongMethodForStream(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/stream", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

func TestHandlePointsFallsBackToInlinePCA(t *testing.T) {
	fake := newFakeQdrant(t)
	app, mux := newTestServer(t, fake)
	app.python = "/nonexistent/python3"

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/points?collection=meta_bridge&limit=2&projection=pca", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var body PointsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode points: %v", err)
	}
	if len(body.Points) != 2 {
		t.Fatalf("points = %d, want 2", len(body.Points))
	}
	if body.Projection == nil || body.Projection.Method != "pca-go-fallback" {
		t.Fatalf("projection = %+v, want pca-go-fallback", body.Projection)
	}
}

func TestHandlePointsFallbackSkipsNonPCAMethods(t *testing.T) {
	fake := newFakeQdrant(t)
	app, mux := newTestServer(t, fake)
	app.python = "/nonexistent/python3"

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/points?collection=meta_bridge&limit=2&projection=umap", nil))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500 without a fallback", rec.Code)
	}
	var body apiErrorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode envelope: %v", err)
	}
	if body.Error.Code != errCodeProjection {
		t.Errorf("code = %q, want %q", body.Error.Code, errCodeProjection)
	}
}

func TestReportProgressPublishesStreamEvent(t *testing.T) {
	fake := newFakeQdrant(t)
	app, _ := newTestServer(t, fake)

	id, ch := app.hub.subscribe("meta_bridge")
	defer app.hub.unsubscribe(id)

	app.reportProgress("meta_bridge", "load-42", loadProgress{Loaded: 1, Limit: 2, Percent: 50, Status: "scrolling"})

	select {
	case raw := <-ch:
		var envelope streamEvent
		if err := json.Unmarshal(raw, &envelope); err != nil {
			t.Fatalf("decode stream event: %v", err)
		}
		if envelope.Type != "load_progress" {
			t.Fatalf("type = %q, want load_progress", envelope.Type)
		}
		payload, ok := envelope.Payload.(map[string]interface{})
		if !ok {
			t.Fatalf("payload type = %T", envelope.Payload)
		}
		if payload["progress_id"] != "load-42" || payload["percent"] != float64(50) {
			t.Errorf("payload = %+v, want progress_id=load-42 percent=50", payload)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for load_progress event")
	}
}

func TestLoadProgressEndpointRemoved(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/load-progress?id=anything", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404 after SSE migration", rec.Code)
	}
}
