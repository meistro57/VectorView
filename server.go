package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

type loadProgressEvent struct {
	ProgressID string `json:"progress_id"`
	Loaded     int    `json:"loaded"`
	Limit      int    `json:"limit"`
	Percent    int    `json:"percent"`
	Status     string `json:"status"`
	Done       bool   `json:"done"`
	Error      string `json:"error,omitempty"`
}

type server struct {
	q           *qdrantClient
	pointsCache projectionCache
	cacheLabels []string
	projections *projectionGuard
	flights     *projectionFlights
	startedAt   time.Time

	defaultCollection string
	maxPoints         int
	qdrantURL         string

	inlinePCAFallbackMax int
	python               string

	hub                      *streamHub
	streamHeartbeatSeconds   int
	streamReplayDefault      int
	streamMaxEventsPerSecond int
	wsPingSeconds            int

	semanticProvider  string
	embedModel        string
	ollamaURL         string
	openrouterBaseURL string
	openrouterKey     string
	openrouterModel   string
	openaiBaseURL     string
	openaiKey         string
	openaiModel       string
}

func buildServer(cfg config) (*server, error) {
	cacheBackends := make([]projectionCache, 0, 2)
	cacheLabels := make([]string, 0, 2)

	diskCache, err := newDiskProjectionCache(cfg.cacheDir, cfg.cacheTTL, cfg.cacheMaxMB<<20)
	if err != nil {
		log.Printf("Disk projection cache disabled: %v", err)
	} else if diskCache != nil {
		cacheBackends = append(cacheBackends, diskCache)
		cacheLabels = append(cacheLabels, diskCache.label())
		log.Printf("Projection cache enabled on disk (%s, ttl=%ds, max=%dMB)", diskCache.dir, cfg.cacheTTLSeconds, cfg.cacheMaxMB)
	}
	if cfg.redisURL != "" {
		redisCache, err := newRedisProjectionCache(cfg.redisURL, cfg.cacheTTL)
		if err != nil {
			log.Printf("Redis projection cache disabled: %v", err)
		} else {
			cacheBackends = append(cacheBackends, redisCache)
			cacheLabels = append(cacheLabels, "redis:"+cfg.redisURL)
			log.Printf("Redis projection cache enabled (%s, ttl=%ds)", cfg.redisURL, cfg.cacheTTLSeconds)
		}
	}

	var pointsCache projectionCache
	switch len(cacheBackends) {
	case 1:
		pointsCache = cacheBackends[0]
	case 2:
		pointsCache = &tieredProjectionCache{backends: cacheBackends}
	}

	streamRecorder, err := newRedisStreamRecorder(cfg.redisURL, cfg.streamRecorderKey, cfg.streamRecorderMaxLen)
	if err != nil {
		log.Printf("Redis stream recorder disabled: %v", err)
	}

	return &server{
		q:                        newQdrant(cfg.qdrantURL, cfg.qdrantKey),
		pointsCache:              pointsCache,
		cacheLabels:              cacheLabels,
		projections:              newProjectionGuard(cfg.projectionSlots, cfg.projectionQueue),
		flights:                  newProjectionFlights(),
		startedAt:                time.Now(),
		defaultCollection:        cfg.defaultCollection,
		maxPoints:                cfg.maxPoints,
		qdrantURL:                cfg.qdrantURL,
		inlinePCAFallbackMax:     cfg.inlinePCAFallbackMax,
		python:                   cfg.python,
		hub:                      newStreamHub(streamRecorder),
		streamHeartbeatSeconds:   cfg.streamHeartbeatSeconds,
		streamReplayDefault:      cfg.streamReplayDefault,
		streamMaxEventsPerSecond: cfg.streamMaxEventsPerSecond,
		wsPingSeconds:            cfg.wsPingSeconds,
		semanticProvider:         cfg.semanticProvider,
		embedModel:               cfg.embedModel,
		ollamaURL:                cfg.ollamaURL,
		openrouterBaseURL:        cfg.openrouterBaseURL,
		openrouterKey:            cfg.openrouterKey,
		openrouterModel:          cfg.openrouterModel,
		openaiBaseURL:            cfg.openaiBaseURL,
		openaiKey:                cfg.openaiKey,
		openaiModel:              cfg.openaiModel,
	}, nil
}

func (s *server) routes() *http.ServeMux {
	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatalf("embed static assets: %v", err)
	}
	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.FS(staticSub)))
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/api/collections", s.handleCollections)
	mux.HandleFunc("/api/stream", s.handleStream)
	mux.HandleFunc("/api/ws", s.handleWS)
	mux.HandleFunc("/api/highlight", s.handleHighlight)
	mux.HandleFunc("/api/points", s.handlePoints)
	mux.HandleFunc("/api/search", s.handleSearch)
	mux.HandleFunc("/api/semantic-search", s.handleSemanticSearch)
	mux.HandleFunc("/api/collections/{collection}/points/{point_id}/similar", s.handleSimilar)
	mux.HandleFunc("/api/collections/{collection}/points/{point_id}/similar-radius", s.handleSimilarRadius)
	return mux
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeJSONError(w, http.StatusMethodNotAllowed, errCodeMethodNotAllowed, "method not allowed")
		return
	}
	setCORS(w)
	writeJSONValue(w, http.StatusOK, buildHealth(r.Context(), s.q, s.cacheLabels, s.projections, s.startedAt))
}

func (s *server) handleCollections(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	names, err := s.q.listCollections(r.Context())
	if err != nil {
		log.Printf("ERROR listCollections: %v", err)
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to list collections", err)
		return
	}
	metas := make([]CollectionMeta, 0, len(names))
	for _, name := range names {
		meta := CollectionMeta{Name: name}
		if m, err := s.q.collectionMeta(r.Context(), name); err == nil {
			meta = m
		} else {
			meta.ProjectionReady = false
			meta.ProjectionNote = "metadata unavailable"
		}
		ready, note := s.q.collectionProjectionStatus(r.Context(), name)
		meta.ProjectionReady = ready
		if note != "" {
			meta.ProjectionNote = note
		}
		metas = append(metas, meta)
	}
	json.NewEncoder(w).Encode(metas)
}

func (s *server) handleStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, errCodeMethodNotAllowed, "method not allowed")
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSONError(w, http.StatusInternalServerError, errCodeInternal, "streaming unsupported")
		return
	}

	collection := strings.TrimSpace(r.URL.Query().Get("collection"))
	heartbeatSeconds, err := strconv.Atoi(strings.TrimSpace(r.URL.Query().Get("heartbeat_seconds")))
	if err != nil || heartbeatSeconds <= 0 {
		heartbeatSeconds = s.streamHeartbeatSeconds
	}
	replayCount, err := strconv.Atoi(strings.TrimSpace(r.URL.Query().Get("replay")))
	if err != nil || replayCount < 0 {
		replayCount = s.streamReplayDefault
	}
	if replayCount > 1000 {
		replayCount = 1000
	}
	maxEventsPerSecond, err := strconv.Atoi(strings.TrimSpace(r.URL.Query().Get("max_events_per_second")))
	if err != nil || maxEventsPerSecond <= 0 {
		maxEventsPerSecond = s.streamMaxEventsPerSecond
	}
	if maxEventsPerSecond <= 0 {
		maxEventsPerSecond = 1
	}

	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	subID, stream := s.hub.subscribe(collection)
	defer s.hub.unsubscribe(subID)

	readyPayload, _ := json.Marshal(map[string]interface{}{"status": "ready", "collection": collection, "replay": replayCount, "max_events_per_second": maxEventsPerSecond})
	if err := writeSSEEvent(w, "stream_ready", readyPayload); err != nil {
		return
	}
	flusher.Flush()

	if replayCount > 0 {
		replayed, err := s.hub.replay(r.Context(), replayCount, collection)
		if err != nil {
			log.Printf("stream replay error: %v", err)
		} else {
			for _, payload := range replayed {
				if err := writeSSEEvent(w, streamEventType(payload), payload); err != nil {
					return
				}
			}
			flusher.Flush()
		}
	}

	heartbeatTicker := time.NewTicker(time.Duration(heartbeatSeconds) * time.Second)
	defer heartbeatTicker.Stop()
	rateTicker := time.NewTicker(time.Second / time.Duration(maxEventsPerSecond))
	defer rateTicker.Stop()
	nextSendAt := time.Now()
	var pendingPayload []byte
	pendingType := "event"

	flushPending := func() bool {
		if len(pendingPayload) == 0 || time.Now().Before(nextSendAt) {
			return true
		}
		if err := writeSSEEvent(w, pendingType, pendingPayload); err != nil {
			return false
		}
		flusher.Flush()
		pendingPayload = nil
		nextSendAt = time.Now().Add(time.Second / time.Duration(maxEventsPerSecond))
		return true
	}

	for {
		select {
		case <-r.Context().Done():
			return
		case payload, ok := <-stream:
			if !ok {
				return
			}
			eventType := streamEventType(payload)
			if time.Now().After(nextSendAt) || time.Now().Equal(nextSendAt) {
				if err := writeSSEEvent(w, eventType, payload); err != nil {
					return
				}
				flusher.Flush()
				nextSendAt = time.Now().Add(time.Second / time.Duration(maxEventsPerSecond))
				continue
			}
			pendingPayload = payload
			pendingType = eventType
		case <-rateTicker.C:
			if !flushPending() {
				return
			}
		case <-heartbeatTicker.C:
			if _, err := io.WriteString(w, ": ping\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func (s *server) handleWS(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSONError(w, http.StatusMethodNotAllowed, errCodeMethodNotAllowed, "method not allowed")
		return
	}
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Duration(s.wsPingSeconds) * time.Second))
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(2 * time.Duration(s.wsPingSeconds) * time.Second))
	})

	go func() {
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				_ = conn.Close()
				return
			}
		}
	}()

	ticker := time.NewTicker(time.Duration(s.wsPingSeconds) * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			deadline := time.Now().Add(5 * time.Second)
			if err := conn.WriteControl(websocket.PingMessage, []byte("ping"), deadline); err != nil {
				return
			}
		}
	}
}

func (s *server) handleHighlight(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	switch r.Method {
	case http.MethodPost:
		var req highlightRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, fmt.Sprintf("invalid request body: %v", err))
			return
		}
		collection := strings.TrimSpace(req.Collection)
		if collection == "" {
			collection = s.defaultCollection
		}
		ids := make([]interface{}, 0, len(req.IDs))
		for _, id := range req.IDs {
			if canonicalPointID(id) == "" {
				continue
			}
			ids = append(ids, id)
		}
		if len(ids) == 0 {
			writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "ids must include at least one point id")
			return
		}
		event := highlightEvent{
			EventID:    strconv.FormatInt(time.Now().UnixNano(), 10),
			Collection: collection,
			IDs:        ids,
			FocusID:    req.FocusID,
			CreatedAt:  time.Now().UnixMilli(),
		}
		if canonicalPointID(event.FocusID) == "" {
			event.FocusID = ids[0]
		}
		highlightEvents.Store(collection, event)
		s.hub.publish("highlight", collection, event)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(event)
	case http.MethodGet:
		collection := strings.TrimSpace(r.URL.Query().Get("collection"))
		if collection == "" {
			collection = s.defaultCollection
		}
		since := strings.TrimSpace(r.URL.Query().Get("since"))
		value, ok := highlightEvents.Load(collection)
		if !ok {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		event, ok := value.(highlightEvent)
		if !ok {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if since != "" && since == event.EventID {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(event)
	default:
		writeJSONError(w, http.StatusMethodNotAllowed, errCodeMethodNotAllowed, "method not allowed")
	}
}

func (s *server) handlePoints(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	collection := r.URL.Query().Get("collection")
	if collection == "" {
		collection = s.defaultCollection
	}
	limit := s.maxPoints
	if ls := r.URL.Query().Get("limit"); ls != "" {
		if lv, err := strconv.Atoi(ls); err == nil && lv > 0 {
			limit = lv
		}
	}
	appendFrom, err := parseAppendFrom(r.URL.Query().Get("append_from"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}
	if appendFrom > limit {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "append_from cannot exceed limit")
		return
	}
	progressID := strings.TrimSpace(r.URL.Query().Get("progress_id"))
	report := func(id string, progress loadProgress) { s.reportProgress(collection, id, progress) }
	vectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))
	projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}
	cacheKey := projectionCacheKey(collection, limit, projectionMethod, vectorName)

	if s.pointsCache != nil {
		if cached, ok, err := s.pointsCache.Get(r.Context(), cacheKey); err == nil && ok && cached != nil {
			if appendFrom > 0 {
				if appendFrom > len(cached.Points) {
					writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "append_from exceeds cached total")
					return
				}
				incremental := PointsResponse{
					Points:       append([]PointData(nil), cached.Points[appendFrom:]...),
					Total:        len(cached.Points),
					Projection:   cached.Projection,
					Incremental:  true,
					AppendFrom:   appendFrom,
					CachedResult: true,
				}
				json.NewEncoder(w).Encode(incremental)
				return
			}
			json.NewEncoder(w).Encode(cached)
			return
		} else if err != nil {
			log.Printf("projection cache read miss/error: %v", err)
		}
	}

	if appendFrom > 0 && projectionMethod == "random" && s.pointsCache != nil {
		baseKey := projectionCacheKey(collection, appendFrom, projectionMethod, vectorName)
		if base, ok, err := s.pointsCache.Get(r.Context(), baseKey); err == nil && ok && base != nil && len(base.Points) == appendFrom {
			deltaRaw, err := s.q.fetchPointsWindow(r.Context(), collection, appendFrom, limit-appendFrom)
			if err == nil {
				deltaResp, err := runProjectionOnPoints(r.Context(), s.python, deltaRaw, projectionMethod, vectorName)
				if err == nil {
					mergedPoints := make([]PointData, 0, len(base.Points)+len(deltaResp.Points))
					mergedPoints = append(mergedPoints, base.Points...)
					mergedPoints = append(mergedPoints, deltaResp.Points...)
					merged := &PointsResponse{Points: mergedPoints, Total: len(mergedPoints), Projection: deltaResp.Projection}
					if err := s.pointsCache.Set(r.Context(), cacheKey, merged); err != nil {
						log.Printf("projection cache write error: %v", err)
					}
					incremental := PointsResponse{
						Points:      append([]PointData(nil), deltaResp.Points...),
						Total:       len(mergedPoints),
						Projection:  deltaResp.Projection,
						Incremental: true,
						AppendFrom:  appendFrom,
					}
					json.NewEncoder(w).Encode(incremental)
					return
				}
				log.Printf("incremental projection fallback (worker): %v", err)
			} else {
				log.Printf("incremental projection fallback (fetch): %v", err)
			}
		}
	}

	log.Printf("→ /api/points collection=%s limit=%d projection=%s vector=%q", collection, limit, projectionMethod, vectorName)
	workerCtx := context.WithoutCancel(r.Context())
	resp, err := s.flights.do(cacheKey, func() (PointsResponse, error) {
		return runGPUPCA(workerCtx, s.python, report, collection, limit, s.qdrantURL, projectionMethod, vectorName, progressID)
	})
	if err != nil {
		if fallback, ok := s.inlineProjection(r.Context(), collection, limit, projectionMethod, vectorName); ok {
			log.Printf("projection worker failed (%v); served inline Go PCA for collection=%s limit=%d", err, collection, limit)
			s.reportProgress(collection, progressID, loadProgress{Loaded: fallback.Total, Limit: limit, Percent: 100, Status: "done", Done: true})
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(fallback)
			return
		}
		s.reportProgress(collection, progressID, loadProgress{Loaded: 0, Limit: limit, Percent: 0, Status: "failed", Done: true, Error: err.Error()})
		log.Printf("ERROR runGPUPCA: %v", err)
		writeProjectionError(w, err)
		return
	}
	if s.pointsCache != nil {
		if err := s.pointsCache.Set(r.Context(), cacheKey, &resp); err != nil {
			log.Printf("projection cache write error: %v", err)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) reportProgress(collection, progressID string, progress loadProgress) {
	if s.hub == nil {
		return
	}
	s.hub.publish("load_progress", collection, loadProgressEvent{
		ProgressID: progressID,
		Loaded:     progress.Loaded,
		Limit:      progress.Limit,
		Percent:    progress.Percent,
		Status:     progress.Status,
		Done:       progress.Done,
		Error:      progress.Error,
	})
}

func (s *server) inlineProjection(ctx context.Context, collection string, limit int, projectionMethod, vectorName string) (*PointsResponse, bool) {
	if projectionMethod != "pca" {
		return nil, false
	}
	if s.inlinePCAFallbackMax <= 0 || limit > s.inlinePCAFallbackMax {
		return nil, false
	}
	fetchCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	raw, err := s.q.fetchPointsWindow(fetchCtx, collection, 0, limit)
	if err != nil {
		log.Printf("inline PCA fallback fetch failed: %v", err)
		return nil, false
	}
	if len(raw) == 0 {
		return nil, false
	}
	points, meta := goPCA3D(raw)
	if len(points) == 0 {
		return nil, false
	}
	if meta == nil {
		meta = &ProjectionMeta{}
	}
	meta.Method = "pca-go-fallback"
	return &PointsResponse{Points: points, Total: len(points), Projection: meta}, true
}

func (s *server) handleSearch(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	collection := r.URL.Query().Get("collection")
	if collection == "" {
		collection = s.defaultCollection
	}
	queryStr := r.URL.Query().Get("q")
	limit := 500
	if lv, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && lv > 0 {
		limit = lv
	}
	projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}
	vectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))

	type matchValue struct {
		Text string `json:"text"`
	}
	type filterMatch struct {
		Key   string     `json:"key"`
		Match matchValue `json:"match"`
	}
	type filterClause struct {
		Should []filterMatch `json:"should"`
	}
	type searchReq struct {
		Filter      filterClause `json:"filter"`
		Limit       int          `json:"limit"`
		WithPayload bool         `json:"with_payload"`
		WithVectors bool         `json:"with_vectors"`
	}

	searchableKeys := []string{
		"text", "content", "chunk_text", "title", "source_title", "summary", "evidence_summary", "source_quote", "question",
		"attachment_filename", "attachment_category", "project", "conversation_id", "session_id", "memory_id", "thought_id",
		"speaker", "memory_kind", "source_type", "strategy", "uncertainty_class", "claims", "concepts", "questions", "tags",
		"sources", "source_id", "file_source", "source_file", "tone", "author", "awakening_phase",
		"TEXT", "CONTENT", "CHUNK_TEXT", "TITLE", "SOURCE_TITLE", "SUMMARY", "EVIDENCE_SUMMARY", "SOURCE_QUOTE", "QUESTION",
		"ATTACHMENT_FILENAME", "ATTACHMENT_CATEGORY", "PROJECT", "CONVERSATION_ID", "SESSION_ID", "MEMORY_ID", "THOUGHT_ID",
		"SPEAKER", "MEMORY_KIND", "SOURCE_TYPE", "STRATEGY", "UNCERTAINTY_CLASS", "CLAIMS", "CONCEPTS", "QUESTIONS", "TAGS",
		"SOURCES", "SOURCE_ID", "FILE_SOURCE", "SOURCE_FILE", "TONE", "AUTHOR", "AWAKENING_PHASE",
	}
	matches := make([]filterMatch, 0, len(searchableKeys))
	for _, key := range searchableKeys {
		matches = append(matches, filterMatch{Key: key, Match: matchValue{Text: queryStr}})
	}

	sreq := searchReq{
		Filter:      filterClause{Should: matches},
		Limit:       limit,
		WithPayload: true,
		WithVectors: true,
	}

	raw, status, err := s.q.do(r.Context(), http.MethodPost,
		"/collections/"+collection+"/points/scroll", sreq)
	if err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "search request failed", err)
		return
	}
	if status >= 400 {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "qdrant rejected the search request", fmt.Errorf("qdrant %d: %s", status, string(raw)))
		return
	}

	var res scrollResult
	if err := json.Unmarshal(raw, &res); err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to decode the qdrant search response", err)
		return
	}

	var pts []QPoint
	if res.Result != nil {
		pts = res.Result.Points
	}

	keySearch := fmt.Sprintf("search:%s:%s:%d:%s:%q", collection, projectionMethod, limit, vectorName, queryStr)
	workerCtx := context.WithoutCancel(r.Context())
	resp, err := s.flights.do(keySearch, func() (PointsResponse, error) {
		return runProjectionOnPoints(workerCtx, s.python, pts, projectionMethod, vectorName)
	})
	if err != nil {
		writeProjectionError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleSemanticSearch(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	sourceCollection := strings.TrimSpace(r.URL.Query().Get("collection"))
	if sourceCollection == "" {
		sourceCollection = s.defaultCollection
	}
	targetCollection := strings.TrimSpace(r.URL.Query().Get("target_collection"))
	if targetCollection == "" {
		targetCollection = sourceCollection
	}
	queryStr := strings.TrimSpace(r.URL.Query().Get("q"))
	if queryStr == "" {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "q is required")
		return
	}
	limit := 120
	if lv, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && lv > 0 {
		limit = lv
	}
	projectionMethod, err := parseProjectionMethod(r.URL.Query().Get("projection"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}
	vectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))

	requestedProvider := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("provider")))
	requestedModel := strings.TrimSpace(r.URL.Query().Get("model"))

	provider := requestedProvider
	model := requestedModel

	detectedProvider, detectedModel, detectedDims := s.q.detectCollectionEmbedding(r.Context(), targetCollection)

	if provider == "" || provider == "auto" {
		if detectedProvider != "" {
			provider = detectedProvider
		} else if s.semanticProvider != "" && s.semanticProvider != "auto" {
			provider = s.semanticProvider
		} else {
			if detectedDims == 3072 {
				provider = "openrouter"
			} else if detectedDims == 1536 {
				provider = "openai"
			} else {
				provider = "ollama"
			}
		}
	}

	if model == "" {
		if detectedModel != "" && (requestedProvider == "" || requestedProvider == "auto" || requestedProvider == detectedProvider) {
			model = detectedModel
		} else {
			switch provider {
			case "openrouter":
				model = s.openrouterModel
			case "openai":
				model = s.openaiModel
			default:
				model = s.embedModel
			}
		}
	}

	var embedding []float64
	switch provider {
	case "openrouter":
		embedding, err = embedQueryWithOpenRouter(r.Context(), s.openrouterBaseURL, s.openrouterKey, model, queryStr, detectedDims)
	case "openai":
		embedding, err = embedQueryWithOpenAI(r.Context(), s.openaiBaseURL, s.openaiKey, model, queryStr, detectedDims)
	case "ollama", "":
		embedding, err = embedQueryWithOllama(r.Context(), s.ollamaURL, model, queryStr)
	default:
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, fmt.Sprintf("unsupported semantic provider: %s", provider))
		return
	}
	if err != nil {
		writeServerError(w, http.StatusBadGateway, errCodeProvider, "embedding provider failed", err)
		return
	}

	queryVector := interface{}(embedding)
	if vectorName != "" {
		queryVector = namedSearchVector{Name: vectorName, Vector: embedding}
	}
	sreq := searchPointsRequest{
		Vector:      queryVector,
		Limit:       limit,
		WithPayload: true,
		WithVectors: true,
	}
	raw, status, err := s.q.do(r.Context(), http.MethodPost,
		"/collections/"+targetCollection+"/points/search", sreq)
	if err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "semantic search request failed", err)
		return
	}
	if status >= 400 {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "qdrant rejected the semantic search request", fmt.Errorf("qdrant %d: %s", status, string(raw)))
		return
	}

	var searchRes struct {
		Result []QPoint `json:"result"`
	}
	if err := json.Unmarshal(raw, &searchRes); err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to decode the semantic search response", err)
		return
	}
	keySemantic := fmt.Sprintf("semantic:%s:%s:%s:%d:%s:%q", targetCollection, provider, model, limit, vectorName, queryStr)
	workerCtx := context.WithoutCancel(r.Context())
	resp, err := s.flights.do(keySemantic, func() (PointsResponse, error) {
		return runProjectionOnPoints(workerCtx, s.python, searchRes.Result, projectionMethod, vectorName)
	})
	if err != nil {
		writeProjectionError(w, err)
		return
	}
	if sourceCollection != targetCollection {
		log.Printf("cross-collection semantic search: source=%s target=%s q=%q hits=%d", sourceCollection, targetCollection, queryStr, len(resp.Points))
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleSimilar(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, errCodeMethodNotAllowed, "method not allowed")
		return
	}
	collection := strings.TrimSpace(r.PathValue("collection"))
	pointID := strings.TrimSpace(r.PathValue("point_id"))
	if collection == "" || pointID == "" {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "collection and point_id are required")
		return
	}

	limit, err := parseSimilarityLimit(r, 12)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}

	selected, err := s.q.getPointByID(r.Context(), collection, pointID)
	if err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to load the selected point", err)
		return
	}
	if selected == nil {
		writeJSONError(w, http.StatusNotFound, errCodeNotFound, "selected point not found")
		return
	}

	queryVectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))
	searchVector, vectorName := pickSearchVector(selected.Vector, queryVectorName)
	if searchVector == nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "selected point has no usable vector")
		return
	}

	// Use the selected point vector directly as the Qdrant similarity query.
	sreq := searchPointsRequest{
		Vector:      searchVector,
		Limit:       limit + 1,
		WithPayload: true,
	}
	raw, status, err := s.q.do(r.Context(), http.MethodPost,
		"/collections/"+collection+"/points/search", sreq)
	if err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "similarity search failed", err)
		return
	}
	if status >= 400 {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "qdrant rejected the similarity search", fmt.Errorf("qdrant %d: %s", status, string(raw)))
		return
	}

	var searchRes searchPointsResult
	if err := json.Unmarshal(raw, &searchRes); err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to decode the similarity response", err)
		return
	}

	selectedKey := canonicalPointID(selected.ID)
	neighbors := make([]similarNeighbor, 0, limit)
	for _, hit := range searchRes.Result {
		if canonicalPointID(hit.ID) == selectedKey {
			continue
		}
		neighbors = append(neighbors, similarNeighbor{ID: hit.ID, Score: hit.Score, Payload: hit.Payload})
		if len(neighbors) >= limit {
			break
		}
	}

	resp := similarResponse{
		SelectedID: selected.ID,
		Collection: collection,
		Limit:      limit,
		VectorName: vectorName,
		Neighbors:  neighbors,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleSimilarRadius(w http.ResponseWriter, r *http.Request) {
	setCORS(w)
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, errCodeMethodNotAllowed, "method not allowed")
		return
	}
	collection := strings.TrimSpace(r.PathValue("collection"))
	pointID := strings.TrimSpace(r.PathValue("point_id"))
	if collection == "" || pointID == "" {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "collection and point_id are required")
		return
	}

	radius, err := parseSimilarityRadius(r, 0.92)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}
	limit, err := parseSimilarityLimit(r, 400)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, err.Error())
		return
	}

	selected, err := s.q.getPointByID(r.Context(), collection, pointID)
	if err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to load the selected point", err)
		return
	}
	if selected == nil {
		writeJSONError(w, http.StatusNotFound, errCodeNotFound, "selected point not found")
		return
	}

	queryVectorName := strings.TrimSpace(r.URL.Query().Get("vector_name"))
	searchVector, vectorName := pickSearchVector(selected.Vector, queryVectorName)
	if searchVector == nil {
		writeJSONError(w, http.StatusBadRequest, errCodeBadRequest, "selected point has no usable vector")
		return
	}

	sreq := searchPointsRequest{
		Vector:         searchVector,
		Limit:          limit + 1,
		WithPayload:    true,
		ScoreThreshold: &radius,
	}
	raw, status, err := s.q.do(r.Context(), http.MethodPost,
		"/collections/"+collection+"/points/search", sreq)
	if err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "radius search failed", err)
		return
	}
	if status >= 400 {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "qdrant rejected the radius search", fmt.Errorf("qdrant %d: %s", status, string(raw)))
		return
	}

	var searchRes searchPointsResult
	if err := json.Unmarshal(raw, &searchRes); err != nil {
		writeServerError(w, http.StatusInternalServerError, errCodeQdrant, "failed to decode the radius response", err)
		return
	}

	selectedKey := canonicalPointID(selected.ID)
	neighbors := make([]similarNeighbor, 0, limit)
	for _, hit := range searchRes.Result {
		if canonicalPointID(hit.ID) == selectedKey {
			continue
		}
		neighbors = append(neighbors, similarNeighbor{ID: hit.ID, Score: hit.Score, Payload: hit.Payload})
		if len(neighbors) >= limit {
			break
		}
	}

	resp := similarResponse{
		SelectedID:       selected.ID,
		Collection:       collection,
		Limit:            limit,
		SimilarityRadius: &radius,
		VectorName:       vectorName,
		Neighbors:        neighbors,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
