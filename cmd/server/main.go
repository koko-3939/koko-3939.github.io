package main

import (
    "log"
    "net/http"
    "my-go-project/graph"
	"my-go-project/graph/generated"
    "github.com/99designs/gqlgen/graphql/handler"
    "github.com/99designs/gqlgen/graphql/playground"
)

func main() {
    srv := handler.NewDefaultServer(generated.NewExecutableSchema(generated.Config{Resolvers: &graph.Resolver{}}))

    http.Handle("/", playground.Handler("GraphQL playground", "/query"))
    http.Handle("/query", srv)

    log.Println("connect to http://0.0.0.0:8080/ for GraphQL playground")
    log.Fatal(http.ListenAndServe("0.0.0.0:8080", nil))
}