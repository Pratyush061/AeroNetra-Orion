# AeroNetra-Orion — A Friendly Guide

**A companion to `README.md`. If the README felt like a wall of technical words, start here. This guide explains what the project is, why it exists, how to run it, how to test it, and what you can do with it — in plain language.**

---

## 1. What is this, in one minute?

AeroNetra-Orion is a small program that watches the world through **two kinds of camera at the same time** and tries to **guess where things are about to move** — so a flying drone can step out of the way *before* a collision, not after.

Think of it as giving a drone the ability to look slightly into the future.

---

## 2. The two kinds of camera

**An ordinary camera** behaves like a person taking photos. It takes 30 pictures every second. If something moves quickly while a photo is being taken, that thing comes out **smeared and blurry** — just like a photo of a running dog. This is a real problem for a fast drone.

**An event camera** is different. It does not take pictures. It only reports *change*. Every time a single spot in the view gets brighter or darker, it sends a tiny instant message: "something changed here, right now." It is fast (millionths of a second), it never blurs, and it works in very bright or very dim light.

> **A helpful way to picture it:** an ordinary camera is someone photographing the room every second. An event camera is a friend who taps your shoulder only when something actually moves. The friend tells you *less*, but what they tell you is *exactly when it happened*.

The blurry photo and the event messages describe **the same moment**, so we can combine them.

---

## 3. The big idea, step by step

Here is the whole pipeline, in everyday words:

1. **Listen to the events.** The event camera says where things are moving.
2. **Repair the blurry photo.** Use those event messages to put the lost sharpness back into the ordinary photo. We now have a clearer picture of the moment.
3. **Spot the objects.** Look at the clearer picture and draw a box around each thing that looks like an object.
4. **Follow each object.** Watch each box across the frames and learn two things: *how fast it is sliding across the view*, and *how quickly it is growing* (a thing heading straight at you grows fast — that is how you know it is getting closer).
5. **Guess the future.** Use what you learned to sketch where each object will be over the next second.
6. **Ask the safety question.** Will any of those future positions land in the corridor the drone is flying towards? If yes, work out **how many seconds until it happens** — the "time to collision" — and raise an alert.

The last two steps are the whole point: **deciding early.**

---

## 4. Why this is interesting

Most drone cameras only answer *"what is here right now?"* By the time a fast object is big enough to be recognised easily, there is very little time left to react.

Orion answers a more useful question: *"what is about to happen, and will it be in my way?"* That extra second of warning is exactly what a drone needs to turn, slow down, or climb.

---

## 5. What is inside

Each part of the project has one job. You do not need to read the code to use it, but here is the map.

| Plain name | Where it lives | What it does |
|---|---|---|
| The event maker | `src/orion/events.py` | Turns ordinary frames into event messages, and back into pictures |
| The scene maker | `src/orion/scene.py` | Draws a pretend aerial view with moving vehicles — so you can try everything without downloading anything |
| The repairer | `src/orion/fusion.py` | Uses events to sharpen the blurry frame |
| The spotter | `src/orion/detect.py` | Finds objects and draws boxes around them |
| The follower | `src/orion/track.py` | Gives each object a name (an ID) and follows it frame by frame |
| The fortune-teller | `src/orion/predict.py` | Guesses where each object will be next |
| The safety officer | `src/orion/ttc.py` | Works out the time to collision and the danger map |
| The conductor | `src/orion/pipeline.py` | Runs all of the above in the right order |
| The command line | `src/orion/cli.py` | The `orion` command you type |

---

## 6. What you need before you start

- A computer running **Windows, macOS or Linux**.
- **Python 3.10 or newer**. Check by typing `python --version` in a terminal.
- About **200 MB of free space** and a few minutes.
- **No graphics card, no internet connection, and no dataset downloads are needed.** The project draws its own pretend scene.

Everything else (the maths and image libraries) is installed automatically in the next step.

---

## 7. Install it — step by step

Open a terminal (Command Prompt or PowerShell on Windows; Terminal on macOS/Linux) and type these lines one at a time.

**Step 1 — get the code**

```bash
git clone https://github.com/Pratyush061/AeroNetra-Orion.git
cd AeroNetra-Orion
```

**Step 2 — make a private workspace (recommended)**

This keeps the project's libraries separate from the rest of your computer.

```bash
python -m venv .venv
```

Now switch it on:

- macOS / Linux: `source .venv/bin/activate`
- Windows: `.venv\Scripts\activate`

You will see `(.venv)` appear at the start of your prompt. That means it worked.

**Step 3 — install the project**

```bash
pip install -e .
```

That is it. The `-e` means "editable" — if you change the code, it takes effect immediately.

**Step 4 — check it worked**

```bash
python -c "import orion; print(orion.__version__)"
```

You should see `0.1.0`.

---

## 8. Run it — and what you will see

```bash
python -m orion.cli demo --out outputs
```

(You can also just type `orion demo` once installed, or `python scripts/run_demo.py`.)

It runs in a couple of seconds and prints something like this:

```
AeroNetra-Orion — demo complete
  frames: 72
  events_total: 113,325
  mean_detections: 9.58
  mean_tracks: 9.19
  critical_alerts: 52
  min_ttc_s: 0.033
  mean_recall: 0.702
  elapsed_s: 2.62
  video: outputs/orion_demo.mp4
```

Then open the `outputs` folder. You will find:

- **`orion_demo.mp4`** — a short video showing the scene with boxes around objects, coloured lines showing where each object is predicted to go, and a small text panel of live numbers.
- **`metrics.json`** — the same summary numbers, saved as a file.

### How to read the numbers

| Number | Plain meaning |
|---|---|
| `frames` | How many pictures were processed |
| `events_total` | How many "something changed" messages the event camera produced |
| `mean_detections` | On average, how many objects were found per picture |
| `mean_tracks` | How many objects were being followed |
| `critical_alerts` | How many times an object was predicted to enter the drone's path soon |
| `min_ttc_s` | The shortest warning time seen — the closest call |
| `mean_recall` | Of the objects that were really there, what fraction we found |
| `mean_precision` | Of the objects we reported, what fraction were real |

> These numbers come from the built-in pretend scene, so treat them as a demonstration of the *machinery*, not a world record. The scene is fixed with a "seed", so if you run it again you get the same result — that is deliberate, and it is how good science is done.

---

## 9. Test it

The project ships with its own checks. Run them like this:

```bash
pip install -e ".[dev]"
pytest -q
```

You should see a row of dots and a line like `50 passed`. Each dot is one automatic check confirming a piece of the maths or logic behaves correctly.

You can also ask it to check its own code style:

```bash
ruff check .
```

---

## 10. Change how it behaves

Everything you might want to adjust lives in one readable file: **`configs/default.yaml`**. Open it in any text editor. For example:

- `scene.frames` — how many pictures to generate.
- `scene.num_vehicles` — how busy the road is.
- `scene.ego_speed` — how fast the drone itself is moving.
- `events.threshold` — how sensitive the event camera is.
- `detect.diff_threshold` — how different a thing must be before it counts as an object.
- `predict.horizon_s` — how far into the future to look.

Then run it with your file:

```bash
python -m orion.cli run --config configs/default.yaml --out outputs
```

You can also peek at the list of real datasets it can talk to:

```bash
orion list-datasets
```

---

## 11. Things to try

- Turn `scene.motion_blur_px` up to 15 and watch the repaired image matter more.
- Set `scene.num_intruders` to 3 and see the alert count climb.
- Turn `events.threshold` down to make the event camera more twitchy.
- Change `predict.horizon_s` to 2.0 and see the alerts fire earlier.

If you want the optional "learning" upgrade (it guesses motion with a small neural network instead of pure physics), install it and train it:

```bash
pip install -e ".[learn]"
python scripts/train_world_model.py --epochs 30 --out outputs/world_model.pt
```

This needs a little more setup and is entirely optional — the project works fully without it.

---

## 12. When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `python: command not found` | Python is not installed | Install Python 3.10+ from python.org, then reopen the terminal |
| `No module named orion` | The install step did not finish | Make sure your `(.venv)` is active, then run `pip install -e .` again |
| The video file is missing | Your machine has no video encoder | Nothing is broken — the numbers and everything else still work |
| `ModuleNotFoundError: cv2` | A library is missing | Run `pip install -e .` again |

---

## 13. Word list

- **Frame** — one picture from the ordinary camera.
- **Event** — one "something changed here" message from the event camera.
- **Fusion** — combining the two, so the blurry picture gets sharp again.
- **Detection** — a box drawn around something the program thinks is an object.
- **Track** — an object being followed over time, with its own name (ID).
- **Prediction** — the program's guess about where an object will be next.
- **Time to collision (TTC)** — how many seconds until an object reaches the drone's path.
- **Occupancy map** — a picture of where things are likely to be soon, drawn as a heat glow.
- **Seed** — a fixed starting number, so the pretend scene is the same every time.

---

*Made by Pratyush Jain. MIT licence — see `LICENSE`. For the technical description, read `README.md`; for the reasoning behind the design, read `docs/innovation.md`.*
