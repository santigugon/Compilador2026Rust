# Documentación: analizador léxico y validación semántica básica

Este proyecto lee el archivo `input.txt`, tokeniza el texto combinando un **autómata finito** y una **capa de expresiones regulares**, fusiona ambos resultados y aplica una **comprobación semántica mínima** sobre el uso de operadores aritméticos. La salida va a la consola: errores léxicos por `stderr`, y la tabla de tokens solo si no hay errores léxicos.

---

## Flujo de ejecución

1. **`read_file::read_input`** — Lee `input.txt` y añade un espacio al final (ayuda a cerrar el último token del autómata).
2. **`automata::match_transitions`** — Recorre el texto carácter a carácter con un autómata; en cada posición puede quedar un token reconocido o `Unknown`. Acumula **incidencias léxicas** (carácter inesperado, literal incompleto, etc.).
3. **`automata::return_used_positions`** — Marca qué posiciones del texto “cubre” el token del autómata (recorriendo desde el final del buffer hacia atrás para saltar huecos `Unknown`).
4. **`utils::regex_match`** — Para cada regla de una lista fija, busca coincidencias en el texto sin solaparse con posiciones ya ocupadas por una coincidencia anterior (el orden de las reglas importa).
5. **`merge_lists::automata_regex_match`** — Por cada índice de carácter: si el autómata tiene prioridad en esa posición **y** el token regex en esa posición no es más largo que el del autómata, se usa el token del autómata; si no, el de la regex. Luego filtra `Unknown`, espacios e indentación, reclasifica identificadores que coinciden con palabras reservadas de Python como **Keyword**, y detecta **secuencias de caracteres no reconocidos** como error léxico (`unknown token`).
6. **`semantic_analysis::validate_operator_operands`** — Comprueba que los operadores del autómata (`+`, `-`, `*`, `**`, `/`, `//`, `%`) tengan operandos válidos a izquierda y derecha.
7. **`main`** — Si hay incidencias léxicas, imprime los problemas y **no** muestra la tabla de tokens; si el léxico está limpio, imprime la tabla. El resultado semántico se informa siempre (OK o lista de errores).

---

## Analizador léxico

### Capa de autómata (`src/transition_matching/automata.rs`)

- Parte del estado **`Q0`**. Cada carácter avanza el estado según `state_match`.
- Reconoce entre otros:
  - Palabras clave **`while`**, **`if`**, **`else`**, **`elif`** (con _lookahead_: si tras `while` viene algo que no sea espacio, salto de línea o `(`, la secuencia puede seguir como **identificador**, p. ej. `whilea`).
  - **Enteros** (`0`–`9`), **flotantes** (entero + `.` + dígitos; un `.` solo al final del archivo genera error de literal incompleto).
  - **Identificadores** (letra o `_` al inicio, luego alfanuméricos o `_`), incluyendo prefijos de palabras clave no completas (`wh`, `ifx`, etc.).
  - **Operadores** reconocidos solo por esta capa: `+`, `-`, `*`, `**`, `/`, `//`, `%`.
  - **Delimitador** `(`.
  - **Cadenas** entre comillas dobles `"..."` (un salto de línea dentro sin cerrar comilla lleva a estado muerto / error).
- Espacios y saltos de línea **reinician** el autómata a `Q0` (no forman token en esta capa).
- Cualquier transición no contemplada va a **estado muerto** y se registra una incidencia léxica.
- Al final del archivo se fuerza un paso extra con `'\n'` para cerrar tokens pendientes.

### Capa de expresiones regulares (`src/regex/utils.rs`)

- Lista ordenada de patrones: muchas **palabras reservadas** y literales de Python (`def`, `return`, `True`, operadores compuestos `<=`, `==`, delimitadores `[]`, `{}`, etc.).
- Cada `find_iter` solo marca posiciones aún libres, de modo que la **primera regla que “gana”** en una zona del texto bloquea solapes posteriores (en la práctica, reglas más específicas o más arriba en la lista tienen prioridad sobre otras que compartan prefijos).

### Fusión (`src/smash/merge_lists.rs`)

- Compara longitud y la bandera `positions_list` para decidir autómata vs. regex en cada posición.
- Tras fusionar, se eliminan tokens `Unknown`, `WhiteSpace` e `Indentation` de la salida final.
- Si un lexema quedó como identificador pero es una **palabra reservada** de la lista interna (`if`, `while`, `and`, …), pasa a categoría **Keyword**.

### Categorías de token (`TokenCategory`)

`Keyword`, `Identifier`, `Integer`, `Float`, `Operator`, `Delimiter`, `Indentation`, `WhiteSpace`, `StringToken`, `Unknown`.

---

## Analizador semántico (parte sencilla)

El módulo `src/semantic_analysis/mod.rs` implementa **`validate_operator_operands`**: una máquina de estados lineal sobre la secuencia de tokens **ya fusionada**.

- **Operando válido**: `Integer`, `Float` o `Identifier`.
- **Límite de expresión** (_boundary_): token `Delimiter` o `Keyword` (se asume que delimita o reinicia contexto, p. ej. `if`, `(`, `:`).
- **Operador** en este análisis: solo los que el autómata clasifica como `Operator` (`+`, `-`, `*`, `**`, `/`, `//`, `%`). Operadores solo-regex (`==`, `+=`, etc.) **no** entran en esta comprobación.

Reglas intuitivas:

1. Tras un operando, se espera un operador o un límite.
2. Tras un operador, debe venir otro operando antes de otro operador o un límite (si no, falta operando derecho).
3. Si aparece un operador cuando aún se esperaba el operando izquierdo (inicio o tras otro operador sin operando intermedio), se reporta falta de operando izquierdo.

Al terminar la secuencia, si quedó un operador esperando su operando derecho, también se reporta error.

**Importante:** la validación semántica **no bloquea** la impresión de tokens; el `main` imprime la tabla si el léxico es correcto, aunque falle la semántica (los errores semánticos van a `stderr`).

---

## Cómo probar

Coloca el texto de prueba en `input.txt` y ejecuta:

```bash
cargo run
```

---

## Ejemplos para `input.txt`

Copia **todo** el contenido de uno de los bloques siguientes en `input.txt` (reemplazando lo que haya), guarda y ejecuta `cargo run`. El programa añade un espacio al final al leer el archivo; conviene que el último token quede delimitado por espacio o salto de línea.

### Entrada correcta (léxico y semántica)

Deberías ver `Semantic analysis: OK` y la tabla de tokens sin `Lexical analysis: failed`.

```text
def calc(x, y):
    if x > 0:
        return x + y
    elif x < 0:
        return x - y
    else:
        pass
    while x != 0:
        x = x - 1
    for i in items:
        total = total + i
    a = 1 + 2 + 3
    b = x * y
    c = z / 2
    d = n % 2
    e = p ** 2
    f = q // 2
    (a + b)
    [a, b]
    True and False

```

### Entrada que falla por semántica

El léxico es válido: se imprime la tabla de tokens, pero en `stderr` aparece `Semantic analysis: failed` (aquí, un `+` al final de `resultado = a +` queda sin operando a la derecha).

```text
def demo(x, y):
    if x > 0:
        return x + y
    elif x < 0:
        return x - y
    else:
        pass
    while x != 0:
        x = x - 1
    for i in items:
        acc = acc + i
    z = 1 + 2 + 3
    w = a * b
    resultado = a +

```

### Entrada que falla por léxico

Aparece `Lexical analysis: failed` con mensajes como `unexpected character '$'` o `unknown token`; **no** se muestra la tabla. El carácter `$` no está contemplado en el autómata como inicio de token válido en ese contexto.

```text
def bad_lex(x):
    a = 1 + 2
    b = x * 3
    c = "invalido
    d = 4
    e = money$here
    f = 3otrosimbolo
    g = $invalido
    return a

```

---

## Casos de prueba sugeridos

### Léxico: entradas que suelen funcionar

| Contenido en `input.txt` | Comportamiento esperado                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `a + b `                 | Tres tokens: identificador, `+` (operador), identificador. Sin errores léxicos.             |
| `42 3.14 `               | Entero y flotante reconocidos por el autómata.                                              |
| `while (x) `             | `while` como **Keyword** (autómata + lookahead), `(` como **Delimiter**.                    |
| `def foo(): `            | `def` suele venir de la **regex** (palabra reservada); el identificador `foo` del autómata. |
| `x == 1 `                | `==` como **Operator** por regex; `1` entero.                                               |

### Léxico: entradas que fallan o generan incidencias

| Contenido en `input.txt`                                 | Qué ocurre                                                                                                                                                  |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `a @ b `                                                 | Carácter `@` no válido en el autómata en ese contexto → **Lexical analysis: failed** (p. ej. `unexpected character '@'`). No se imprime la tabla de tokens. |
| `"hola` + salto de línea                                 | Cadena sin cerrar / carácter inválido en cadena → incidencias léxicas; puede aparecer además `unknown token` en la fusión.                                  |
| `1. ` al final (solo punto tras dígitos sin más dígitos) | Tras el espacio forzado al leer el archivo, puede reportarse literal flotante incompleto según el estado al EOF.                                            |

### Semántica: entradas válidas

| Contenido en `input.txt` | Resultado                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `a + b `                 | `Semantic analysis: OK`.                                                                                       |
| `1 + 2 + 3 `             | OK encadenando sumas.                                                                                          |
| `if a + b : `            | Tras `if` (Keyword) el analizador reinicia expectativa; `a + b` se valida como subexpresión; suele dar **OK**. |

### Semántica: entradas inválidas (mensajes en consola)

| Contenido en `input.txt` | Resultado típico                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| `a + `                   | Error: el `+` debe ir seguido de entero, flotante o identificador.                                        |
| `+ a `                   | Error: el `+` debe ir **precedido** por operando.                                                         |
| `a + + b `               | Dos errores: el primer `+` sin operando a la derecha “inmediato”; el segundo sin operando a la izquierda. |

En los casos solo semánticos, la tabla de tokens **sí** se muestra si el léxico es correcto; revisa también las líneas `Semantic analysis: failed.`.

---

## Resumen

| Componente       | Rol                                                                                                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Autómata         | Tokenización carácter a carácter de números, identificadores, subconjunto de palabras clave, operadores aritméticos básicos, strings y `(`; errores explícitos. |
| Regex            | Refina y añade tokens al estilo Python (palabras clave extra, operadores compuestos, delimitadores).                                                            |
| Fusión           | Elige autómata o regex por posición, limpia ruido y unifica palabras reservadas.                                                                                |
| Semántica simple | Solo verifica vecinos de `+ - * ** / // %` en la lista final de tokens.                                                                                         |

Para ampliar el lenguaje reconocido habría que extender las transiciones del autómata, la lista de regex y, si se desea, las reglas en `validate_operator_operands`.
