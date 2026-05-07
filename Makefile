.PHONY: run build tidy

run:
	go run .

build:
	go build -o vectorview .

tidy:
	go mod tidy

install-dep:
	go get github.com/joho/godotenv
