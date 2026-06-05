# Mini-Triton Parser

Parser para un subconjunto de Triton con:

- analisis lexico
- parser recursive descent
- AST
- casos validos e invalidos

## Uso

```bash
cargo run -- input.txt
```

Tambien puedes ejecutar la suite incluida:

```bash
cargo run -- --run-tests
```

## Reglas principales

- un solo kernel por archivo
- `@triton.jit` obligatorio
- bloques con `{}` y sentencias con `;`
- solo asignacion y sentencia de expresion
- argumentos solo posicionales
- unica anotacion permitida: `tl.constexpr`

## Salida

- `VALIDO` + AST
- `INVALIDO` + mensaje con posicion
