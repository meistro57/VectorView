package main

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
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
			var req pointsByIDRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			matches := make([]QPoint, 0, len(req.IDs))
			for _, want := range req.IDs {
				for _, point := range fake.points {
					if canonicalPointID(point.ID) == canonicalPointID(want) {
						matches = append(matches, point)
					}
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"result": matches})
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

func TestDecodeVectorParams(t *testing.T) {
	cases := []struct {
		name     string
		raw      string
		wantSize uint64
		wantOK   bool
	}{
		{"single", `{"size":3072,"distance":"Cosine"}`, 3072, true},
		{"named", `{"text":{"size":3072,"distance":"Cosine"},"image":{"size":512,"distance":"Dot"}}`, 3072, true},
		{"named-majority", `{"a":{"size":8,"distance":"Dot"},"b":{"size":8,"distance":"Dot"},"c":{"size":4,"distance":"Dot"}}`, 8, true},
		{"sparse-only", `{"sparse":{"index":{"on_disk":false}}}`, 0, false},
		{"null", `null`, 0, false},
		{"empty", ``, 0, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			size, _, ok := decodeVectorParams(json.RawMessage(c.raw))
			if ok != c.wantOK || size != c.wantSize {
				t.Fatalf("decodeVectorParams(%s) = (%d, ok=%v), want (%d, ok=%v)", c.raw, size, ok, c.wantSize, c.wantOK)
			}
		})
	}
}

func TestTruncateTextIsRuneSafe(t *testing.T) {
	if got := truncateText("héllo wörld", 5); got != "héllo…" {
		t.Errorf("truncateText = %q, want %q", got, "héllo…")
	}
	if got := truncateText("short", 100); got != "short" {
		t.Errorf("truncateText = %q, want short", got)
	}
	if got := truncateText("anything", 0); got != "anything" {
		t.Errorf("truncateText with 0 budget = %q, want unchanged", got)
	}
}

func TestTrimPointsResponse(t *testing.T) {
	long := strings.Repeat("doc ", 200)
	response := PointsResponse{
		Points: []PointData{
			{ID: "p1", Payload: map[string]interface{}{"document": long, "score": 0.5}},
			{ID: "p2", Payload: map[string]interface{}{"text": "short", "tags": []interface{}{long, "ok"}}},
			{ID: "p3", Payload: nil},
		},
		Total: 3,
	}

	trimmed := trimPointsResponse(response, 400)

	if trimmed.Points[0].Payload["score"] != 0.5 {
		t.Error("numeric payload values must survive trimming")
	}
	if doc, _ := trimmed.Points[0].Payload["document"].(string); len([]rune(doc)) != 401 {
		t.Errorf("expected 400 runes plus ellipsis, got %d", len([]rune(doc)))
	}
	if !trimmed.Points[0].PayloadTruncated || !trimmed.Points[1].PayloadTruncated {
		t.Error("expected PayloadTruncated on trimmed points")
	}
	if trimmed.Points[2].PayloadTruncated {
		t.Error("nil payload must not be marked truncated")
	}
	if tags, _ := trimmed.Points[1].Payload["tags"].([]interface{}); len(tags) != 2 {
		t.Error("array length must be preserved")
	}

	untrimmed := trimPointsResponse(response, 0)
	if untrimmed.Points[0].PayloadTruncated {
		t.Error("zero budget must disable trimming")
	}

	if response.Points[0].Payload["document"] != long {
		t.Error("trimPointsResponse must not mutate the input response")
	}
}

func TestCompressionMiddleware(t *testing.T) {
	handler := compressionMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/stream":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: ping\n\n")
		case "/empty":
			w.WriteHeader(http.StatusNoContent)
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"ok":true,"body":"`+strings.Repeat("x", 400)+`"}`)
		}
	}))

	t.Run("compresses when accepted", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/collections", nil)
		req.Header.Set("Accept-Encoding", "gzip")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Header().Get("Content-Encoding") != "gzip" {
			t.Fatalf("Content-Encoding = %q, want gzip", rec.Header().Get("Content-Encoding"))
		}
		reader, err := gzip.NewReader(rec.Body)
		if err != nil {
			t.Fatalf("gzip reader: %v", err)
		}
		decoded, err := io.ReadAll(reader)
		if err != nil {
			t.Fatalf("read gzip body: %v", err)
		}
		if !strings.Contains(string(decoded), `"ok":true`) {
			t.Errorf("decoded body = %q", decoded)
		}
		if rec.Header().Get("Content-Length") != "" {
			t.Error("compressed responses must not advertise Content-Length")
		}
	})

	t.Run("passes through without accept-encoding", func(t *testing.T) {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/collections", nil))
		if rec.Header().Get("Content-Encoding") != "" {
			t.Error("expected uncompressed response")
		}
		if !strings.Contains(rec.Body.String(), `"ok":true`) {
			t.Errorf("body = %q", rec.Body.String())
		}
	})

	t.Run("skips event streams", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/stream", nil)
		req.Header.Set("Accept-Encoding", "gzip")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Header().Get("Content-Encoding") != "" {
			t.Error("SSE responses must not be compressed")
		}
		if !strings.Contains(rec.Body.String(), "data: ping") {
			t.Errorf("stream body = %q", rec.Body.String())
		}
	})

	t.Run("skips bodyless responses", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/empty", nil)
		req.Header.Set("Accept-Encoding", "gzip")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusNoContent {
			t.Fatalf("status = %d, want 204", rec.Code)
		}
		if rec.Header().Get("Content-Encoding") != "" {
			t.Error("204 responses must not be compressed")
		}
	})
}

func TestCollectionCatalogCachesAndReportsNamedVectors(t *testing.T) {
	var metaRequests int32
	qdrantStub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/collections":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"collections": []map[string]string{{"name": "named"}, {"name": "plain"}},
				},
			})
		case r.URL.Path == "/collections/named":
			atomic.AddInt32(&metaRequests, 1)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"points_count": 42,
					"config": map[string]interface{}{
						"params": map[string]interface{}{
							"vectors": map[string]interface{}{
								"text":  map[string]interface{}{"size": 3072, "distance": "Cosine"},
								"image": map[string]interface{}{"size": 512, "distance": "Dot"},
							},
						},
					},
				},
			})
		case r.URL.Path == "/collections/plain":
			atomic.AddInt32(&metaRequests, 1)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"result": map[string]interface{}{
					"points_count": 7,
					"config": map[string]interface{}{
						"params": map[string]interface{}{
							"vectors": map[string]interface{}{"size": 768, "distance": "Cosine"},
						},
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer qdrantStub.Close()

	app := &server{
		q:               newQdrant(qdrantStub.URL, ""),
		collectionCache: &collectionCatalogCache{ttl: time.Minute},
		projections:     newProjectionGuard(2, time.Second),
		flights:         newProjectionFlights(),
	}

	metas, err := app.collectionCatalog(context.Background())
	if err != nil {
		t.Fatalf("collectionCatalog: %v", err)
	}
	if len(metas) != 2 {
		t.Fatalf("metas = %d, want 2", len(metas))
	}
	byName := map[string]CollectionMeta{}
	for _, meta := range metas {
		byName[meta.Name] = meta
	}
	if byName["named"].VectorSize != 3072 {
		t.Errorf("named vector size = %d, want 3072", byName["named"].VectorSize)
	}
	if byName["plain"].VectorSize != 768 {
		t.Errorf("plain vector size = %d, want 768", byName["plain"].VectorSize)
	}
	if !byName["named"].ProjectionReady || byName["named"].ProjectionNote != "ok" {
		t.Errorf("named meta = %+v, want ready", byName["named"])
	}
	if got := atomic.LoadInt32(&metaRequests); got != 2 {
		t.Errorf("metadata requests = %d, want 2 (no vector scroll)", got)
	}

	if _, err := app.collectionCatalog(context.Background()); err != nil {
		t.Fatalf("second collectionCatalog: %v", err)
	}
	if got := atomic.LoadInt32(&metaRequests); got != 2 {
		t.Errorf("cached catalog made %d extra requests", got-2)
	}
}

func TestHandlePointDetail(t *testing.T) {
	fake := newFakeQdrant(t)
	_, mux := newTestServer(t, fake)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/collections/meta_bridge/points/p2", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", rec.Code, rec.Body.String())
	}
	var body pointDetailResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode detail: %v", err)
	}
	if body.Payload["text"] != "beta document" {
		t.Errorf("payload = %+v", body.Payload)
	}
}
