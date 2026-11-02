.PHONY: run build tidy test python-test fmt vet docker-build docker-up docker-down

run:
	go run .

build:
	go build -o vectorview .

tidy:
	go mod tidy

install-dep:
	go get github.com/joho/godotenv

test:
	go test ./...

python-test:
	venv/bin/python3 -m unittest discover -s tests

fmt:
	gofmt -l .

vet:
	go vet ./...

docker-build:
	docker build -t vectorview:local .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
