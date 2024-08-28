# ベースイメージを指定（Alpineベースの軽量イメージを使用）
FROM golang:1.22.6-alpine

# ワーキングディレクトリを設定
WORKDIR /app

# Goモジュールの依存関係を解決
COPY go.mod go.sum ./
RUN go mod download

# アプリケーションのソースコードをコピー
COPY . .

# アプリケーションをビルド
RUN go build -o server cmd/server/main.go

# アプリケーションを実行
CMD ["/app/server"]