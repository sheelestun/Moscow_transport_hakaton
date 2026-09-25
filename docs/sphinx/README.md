# Sphinx documentation

Документация по ML-модулю проекта «Предиктор задержек Москвы».

## Сборка

```bash
cd docs/sphinx
pip install sphinx sphinx-rtd-theme
make html
```

Результат — `docs/sphinx/_build/html/index.html`. Открыть в браузере:

```bash
xdg-open _build/html/index.html   # Linux
open _build/html/index.html       # macOS
```

## Очистка

```bash
make clean
```
