# Cómo arrancar la demo

Abrir 3 terminales (Cmd+T para nuevas pestañas).

---

## Terminal 1 — Ollama (modelo de IA)

```bash
ollama serve
```

Déjala corriendo, no la cierres.

---

## Terminal 2 — API

```bash
cd ~/proyectos/agente-credito
source ~/entornos/datascience/bin/activate
uvicorn api:app --reload
```

Espera a ver: `Uvicorn running on http://127.0.0.1:8000`
Déjala corriendo, no la cierres.

---

## Terminal 3 — ngrok (túnel público)

```bash
ngrok http 8000
```

Copia la URL que aparece junto a `Forwarding` y agrégale `/static/index.html`:

```
https://<lo-que-salga>.ngrok-free.dev/static/index.html
```

Esa es la URL que compartes.

---

## Verificar que todo funciona

```bash
curl http://localhost:8000/health
```

Debe responder: `{"status":"ok",...}`

---

## Checklist antes de la presentación

- [ ] Terminal 1: Ollama corriendo
- [ ] Terminal 2: API corriendo
- [ ] Terminal 3: ngrok corriendo
- [ ] Probaste la URL tú mismo
- [ ] Tienes los datos listos (botón "Cargar Demo" en el frontend)

---

## Notas

- Si cierras ngrok y lo vuelves a abrir, la URL cambia — avisa a los participantes.
- La primera vez que alguien abre la URL de ngrok aparece una pantalla de advertencia, solo hacer clic en "Visit Site".
- No cierres ninguna de las 3 terminales mientras dure la demo.
