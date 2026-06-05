# Documentacion Tecnica: Mini-Triton Parser

Este proyecto implementa el flujo solicitado para Mini-Triton:

1. Analisis lexico por automata + reglas regex.
2. Parsing recursive descent.
3. Construccion de AST.
4. Reporte de `VALIDO` o `INVALIDO` con mensajes y posicion de error.

## Flujo general

- `src/transition_matching/automata.rs`
  Reconoce identificadores, enteros, flotantes, operadores aritmeticos y parte de los delimitadores.
- `src/regex/utils.rs`
  Completa tokens reservados y delimitadores que el automata no cubre por si solo.
- `src/smash/merge_lists.rs`
  Fusiona ambos resultados y conserva la posicion inicial de cada token.
- `src/parser.rs`
  Implementa la gramatica Mini-Triton con descenso recursivo y construye el AST.
- `src/ast.rs`
  Define nodos como `Program`, `Kernel`, `Assign`, `ExprStmt`, `BinaryOp`, `Call`, `Name` y `Number`.
- `src/main.rs`
  Ejecuta el analisis completo sobre un archivo o sobre la suite de pruebas.

## Restricciones soportadas

- Exactamente un kernel por archivo.
- Decorador obligatorio `@triton.jit`.
- Bloques con `{}`.
- Sentencias terminadas en `;`.
- Solo asignaciones y sentencias de expresion.
- Precedencia `()` > llamadas/nombres/numeros > `*` `/` > `+` `-`.
- Llamadas `id(args)`, `tl.id(args)` e `id.id(args)`.
- Argumentos solo posicionales.
- El unico tipo permitido en parametros anotados es `tl.constexpr`.

## Ejecucion

```bash
cargo run -- input.txt
```

O usando el binario con el nombre pedido por la tarea:

```bash
cargo build
./target/debug/mini_triton_parser input.txt
```

Para correr la suite incluida:

```bash
cargo run -- --run-tests
```

## Salida esperada

- Si el archivo es valido: imprime `VALIDO` y el AST.
- Si es invalido: imprime `INVALIDO` y el error con posicion.

Ejemplo de mensaje:

```text
[ParseError at char 25] Se esperaba 'tl' pero llego 'foo' (Identifier)
```
