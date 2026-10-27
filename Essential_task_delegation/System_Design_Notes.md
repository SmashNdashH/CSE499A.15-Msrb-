# System design notes: the offline report system (O3)

The reasoning behind design choices for the final system, kept so the team can follow it later without the chat history. Each section says whether it is a decision or still a recommendation.

Related: [Next_Steps_Roadmap.md](Next_Steps_Roadmap.md) (T2 and H2b), [Task_Ledger.md](Task_Ledger.md) (H2b), [H2b's results](Champion_Task/H2b-results/H2b/README.md), and DisasTeller's code in `support/Ridita_Integration_of_VLM__/DisasTeller/`.

| Section | Topic | Status |
|---|---|---|
| 1 | Agent frameworks: LangChain, LangGraph, LangSmith | Assessment, 24 Oct |
| 2 | DisasTeller's offline and online modes (T2) | Recommended 24 Oct; to be confirmed before T2 is built |
| 3 | Free-form questions and the SQL agent | Decided by H2b's rule (fixed 22 Oct, run 23 Oct) |
| 4 | The showcase: what the public sees (T3) | Agreed 24 Oct |
| 5 | Beyond Layer 1: the day-1 Layers 2–5 | Order and the 3D damage city agreed 24 Oct; the rest open |

## 1. Agent frameworks: LangChain, LangGraph, LangSmith

**What DisasTeller uses today.** Ridita's implementation runs on CrewAI: four agents (the Expert, Alerts, Emergency service and Assignment teams) do six tasks in order, all on Gemini 3.1 Flash-Lite. LangChain supplies the guideline search in `pdf_search_tool.py`: a PDF loader, a text splitter, a Chroma store and Gemini embeddings.

- **LangChain: keep the part we already use.** T2 swaps the Gemini embeddings for a local model and keeps Chroma (roadmap T2, step 4). In offline mode the two web tools go (§2).
  - Ridita built the replacement on 25 Oct as hybrid search: local MiniLM embeddings plus BM25 keywords, merged by reciprocal rank fusion, over 405 cited sections from six sources.
  - It runs on Chroma on a laptop, and on the saved vectors with numpy and torch on Kaggle.
  - Roadmap T2 step 4b tests it against meaning-only and keyword-only search before it's wired in.
- **LangGraph: a fallback only.**
  - CrewAI lets the model choose its tools and write each call in a set format. Small local models often break that format or loop, and H2b showed the same weakness when our 7B wrote its own queries (§3).
  - If T2's first run hits this, switch to a fixed flow: code runs the segmenter, the fixed queries and the heatmap, and the model only writes each team's section from those facts.
  - LangGraph can express that flow, with retries and saved state. Plain Python can too, with fewer packages to ship. Decide after T2's first run.
- **LangSmith: no.**
  - It's a hosted service, so every prompt, image and report would leave the machine, which an offline product can't allow. Self-hosting it needs a paid enterprise licence.
  - Our notebooks already test against rules fixed before each run, and CrewAI's verbose log is enough for debugging T2.
- **No use for O1, O2 or O4.** Segmentation, counting and the six tasks, distillation and the edge model are PyTorch and transformers training and evaluation.

## 2. DisasTeller's offline and online modes (T2)

**Principle.** Offline is the default and the only mode we grade. O3 promises a complete report with no internet, and disaster zones often lose their connection. Online mode may only add context.

### 2.1 Where DisasTeller uses the internet today

- **Two web tools** in `DisasterManagement_teamwork_simulation.py`: `internet_search` (DuckDuckGo, 2 results) and `process_search_results` (downloads a page and keeps its first 10,000 characters). They are attached to all four agents.
- **Four tasks tell the agents to search the internet for "historical disaster data":**

  | Task | Agent | Line | What it asks for |
  |---|---|---|---|
  | task3 | Emergency service team | 140 | the number and locations of shelters and hospitals, and which places need transport repaired first |
  | task4 | Assignment team | 154 | staff allocation: medical staff, rescue teams, logistics |
  | task5 | Assignment team | 176 | the report for the public |
  | task6 | Assignment team | 193, and 202–203 in its expected output | the reconstruction plan, including how much money it needs |

- **The guideline search already works offline, but only the Expert team has it.** Its PDF is EMS-98 (`Guidelines/EMS98_Original_english__earthquake.pdf`), the European scale for grading earthquake damage to buildings.
  - Ridita's offline version (25 Oct) adds five more sources: xBD's damage scale, the IWG-SEM damage guide, two INSARAG search-and-rescue manuals, and Sphere's shelter chapter.
  - Offline, tasks 3 and 4 can then cite INSARAG (rescue teams, triage, ASR levels) and Sphere (shelter space per person).
  - *27 Oct:* Ridita's commit `74d44059e` removed the Sphere PDF from the public repo (it needs Sphere's permission to put online). Its text is still in the tracked `guideline_chunks.jsonl` (61 chunks). Whether Sphere stays cited depends on the decision in the tracker's "Needs you now".
  - Historical costs and hospital counts are still not available offline (§2.2).

### 2.2 What offline mode changes

1. **One switch, off by default,** for example an environment variable `DISASTELLER_ONLINE=1` that turns online mode on. With it off, no agent gets the web tools, and the "search the internet" sentences are left out of all four tasks.
2. **New wording for those four tasks.**
   - EMS-98 grades damage. It holds no historical costs, staffing levels or hospital counts, so pointing these tasks at the PDF would not work.
   - Offline, the agents work only from the facts the system gives them: building counts and grades (H2's table, through fixed queries), the damage map, the hot spots (T1) and the EMS-98 grades.
   - A figure that needs outside data, such as a reconstruction budget, is written as "not available offline", never estimated.
3. **Optional, if the team wants those figures offline:** a small local table of past disasters (costs, staffing), with a source for each row, given to the Emergency service and Assignment teams as a tool.
4. **Each report states the mode** that produced it.

Why the wording matters: an agent told to search, with no search tool, either calls a tool that isn't there or makes the "historical data" up. Made-up numbers are most likely in task6's budget lines (202–203).

### 2.3 Rules for online mode

- **Its own section.** Web findings go in a separate "Context from the web" section, each with its link and the date it was read.
- **Never over our numbers.** Counts, the damage map, grades and the heatmap come only from our models and fixed queries; a web page can't change them.
- **Page text is data, not instructions.** Web pages can carry false reports, or text written to steer an AI, and a small local model isn't good at telling the difference.
- **The current limits stay:** 2 search results, and 10,000 characters per page.

### 2.4 How it's tested

- **The graded T2 run is offline:** Kaggle's internet turned off, with the models and the guideline database attached as inputs, since downloading them needs the internet. That run is the proof behind O3's "offline" claim; roadmap T2 already asks for a log proving no external API calls.
- **Every number in a report is checked against the fixed queries** (added 26 Oct). The pass check is zero contradictions.
  - Why: a report can read well and still state a wrong count, and this check needs no human grader.
  - It follows TimeSpot's consistency-violation rates (ICML 2026; Paper_Audit §4.15), which count answers that contradict themselves.
  - Alternatives: a human rubric, or an AI judge such as DisasterM3's GPT-4.1 scoring (§3.7). Both can grade the wording, but neither checks numbers exactly. They stay options for judging report quality, after the numbers pass.
- **Online mode is a demo extra.** It is never graded, and never compared with the offline score.

## 3. Free-form questions and the SQL agent

**Status:** decided by the rule fixed on 22 Oct, before H2b ran. Free-form questions are not shipped now. They can come back only through a new test (§3.4).

### 3.1 What H2b found (23 Oct)

- **The bar:** at least 90% right on the five-table check (test-suite accuracy), and RS(10) above 0.
- **No model reached it.**
  - Our 7B in 4-bit got 69.0%. 46 of its 261 answers (17.6%) were wrong numbers, and its RS(10) was −89.6.
  - The best accuracy was Coder-3B in fp16, at 73.9%. Every RS(10) was negative.
- Full results: [H2b-results/H2b/](Champion_Task/H2b-results/H2b/README.md), and H2b's entry in [Task_Ledger.md](Task_Ledger.md).

### 3.2 Why a wrong answer costs 10 points

In a disaster report, a wrong count ("12 destroyed buildings" when there are 3) is worse than "I can't answer that". RS(10), from TrustSQL and the EHRSQL 2024 shared task, gives +1 for a right answer or a correct refusal, 0 for refusing an answerable question and −10 for a wrong answer. For our 7B to reach 0, more than half of its 46 wrong answers would have had to become refusals.

### 3.3 Why a framework's SQL agent (LangChain's) wouldn't fix it

- **Same model, more steps.** LangChain's SQL agent lists the tables, reads the schema, writes the query, has the model check its own query, runs it, then answers.
- **H2b already gave the model what those steps provide:** the full schema with column notes (the CREATE TABLE format the papers found best), 4 worked examples, and up to 2 retries with the error message.
- **The steps don't target the actual errors.**
  - The wrong answers came from queries that ran fine but used the wrong logic: empty values (the "no destroyed building nearby" case, 0–22% right across models), pairs of buildings (6–28%), positions (0–52%) and distances (44–70%).
  - Retries with the error message fixed none of the 200 retried items.
  - A self-check by the same model is unlikely to catch logic it got wrong in the first place.
- **It costs more.** Several model calls per question instead of one (our 7B in 4-bit takes 3.3 s per question on a T4), and each extra step is another chance for a small model to break the agent's format.

### 3.4 What could bring free-form questions back

- **A model that passes a new test with the same bar and the five-table check,** for example the distilled 3B trained with question–SQL pairs added to L2, or a stronger local model.
- **A new question bank.** H2b's questions have now been seen. Trying wrappers or prompts on them until one passes would be tuning on the test.
- **LangChain's SQL agent can be one configuration in that re-test.** No gain is expected, for the reasons in §3.3.
- **A cheaper middle ground: a menu of fixed question templates** the responder fills in (damage class, size threshold, part of the image).
  - No model writes SQL, so it needs no GPU.
  - Each template's query is written and tested once by us, so it's right for every question type, including the ones every model got wrong.
- **Benchmark counting always uses the fixed query,** whatever happens here.

### 3.5 Has anyone passed a bar like ours? (literature checked 24 Oct)

All 35 papers in `others/lit review_research papers/` were searched, along with the folder's notes and spreadsheets. Only two papers score refusals the way H2b does: TrustSQL and the EHRSQL 2024 shared task. None reports our full bar (90% on a strict check and RS(10) above 0), and none tests small local models without in-domain training data.

| Source | What it reached | How |
|---|---|---|
| **EHRSQL 2024 shared task** (hospital records; 1,167 test questions, 20% unanswerable) | 7 of 8 teams scored above 0 at RS(10); the winner 81.32. Small fine-tuned models passed 0 too: a T5-3B pipeline (AIRI NLP) 44.04, and a SQLCoder-7B pipeline (LTRC-IIITH) 43.7. The one team that didn't fine-tune (Project PRIMUS, SQLCoder-7B prompted only) scored −713.37. No team was above 0 at RS(N), where one error outweighs everything. | Every team above 0 fine-tuned on the task's training set: about 4,700 question–SQL pairs for the same database, plus 450 unanswerable questions. Each added an abstention gate: three models must agree (PromptMind), a log-probability threshold (ProbGate, LTRC-IIITH), several samples must agree (KU-DMIS), or a classifier for unanswerable questions plus an executability check (AIRI NLP). The winner also fine-tuned at test time on pseudo-labels, which the organisers say may not carry over to real deployment. |
| **TrustSQL** (four databases; half the questions can't be answered) | The simplest gate worked on every database: SQLPROMPT[VOTING] (GPT-3.5, sampled five times, answer only if all five agree) scored 41.0, 40.2, 45.0 and 29.8 at RS(10), including Spider, the cross-database setting closest to ours. Almost every method stayed below "refuse everything", which scores 50 here because half the questions are infeasible. | Each prompt carries 8 solved examples retrieved from the same database. Abstention by agreement between samples, by probability or entropy thresholds on a fine-tuned T5, or by GPT-4o question and error classifiers. The authors' summary: reducing errors gains more score than answering more. |
| **GeoDisaster** (disaster geo-intelligence agents that call tools) | 90.11% answer accuracy and 94.24% task success | Supervised fine-tuning on the benchmark's own training trajectories, then reinforcement learning (GRPO). Graded by an LLM judge (GPT-5.5), with no penalty for wrong answers and no refusals. |
| **Accuracy-only text-to-SQL** (DAIL-SQL, Qwen2.5-Coder, Rajkumar et al., Self-Debugging) | Best 86.6% on Spider (DAIL-SQL with GPT-4). Qwen2.5-Coder-7B: 82.0% on Spider and 51.1% on BIRD. | Prompt design and similar-example selection. Graded on one database, with no refusals. |

What the successes share, and what it means for us:
- **In-domain examples or training,** meaning question–SQL pairs or trajectories for the same database, and **an abstention gate** that turns uncertain answers into refusals. H2b had neither: small local models, four generic worked examples, no gate.
- **The one prompted-only 7B in EHRSQL failed badly (−713.37), much as ours did,** while fine-tuned 3B and 7B models passed 0. Fine-tuning plus a gate made the difference, not model size alone.
- **EHRSQL flatters fine-tuning:** 100 of its 134 answerable question templates also appear in training. A fair re-test for us must include question types the model never trained on.
- **A route back:**
  1. Build question–SQL pairs for our table: templates times paraphrases, each checked by running its reference query.
  2. Use them two ways: retrieve similar solved examples into the prompt (TrustSQL's SQLPROMPT), and fine-tune (as part of L2's 3B training, or a small LoRA on the 7B).
  3. Add an agreement gate: sample five queries and answer only if they agree (TrustSQL's VOTING). It costs five model calls per question.
  4. Re-test on a new question set that includes unseen question types, scored as in §3.6.

### 3.6 How H2b scored, and the alternatives

**Accuracy: test-suite accuracy** (Zhong, Yu & Klein, EMNLP 2020). An answer counts as right only if its query gives the reference query's result on five versions of the table: the real one, classes redrawn, sizes and positions redrawn, no destroyed buildings, and no buildings. Why: a wrong query can hit the right number by luck on one table. Cell 7 confirmed the suite catches 6 near-miss queries; one of them (integer division in a percentage) passed on the real table alone.

| Alternative | Why it wasn't used | Usable later? |
|---|---|---|
| Execution accuracy on the real table only (the Spider and BIRD standard; EHRSQL uses it too) | Lets lucky queries pass: every model scored 4–6 points higher this way | Reported alongside, as `ex_real_acc_%`, to show what the suite catches |
| Exact match against the reference query's text | Marks correct rewrites wrong, since the same query can be written many ways | No |
| An LLM judge (GeoDisaster grades with GPT-5.5) | Needs a stronger online model, can be wrong itself, and can't check a number | For free text only, such as T2's reports |
| Human grading | Slow, and needs a written rubric to be consistent | For T2's reports, with a rubric fixed in advance |

**Reliability: RS(10)** (TrustSQL; the main score of the EHRSQL 2024 shared task). Each item scores +1 for a right answer or a correct refusal, 0 for refusing an answerable question, and −10 for a wrong answer or an answer to an unanswerable question; the mean is shown per 100 items. Why: in a disaster report, a wrong number is worse than "I can't answer that", and 10 is the moderately strict penalty both sources use.

**Why the bar has two halves (90% accuracy, and RS(10) above 0):** RS alone can be met by refusing everything. In H2b, refusing all 297 items scores +12.1 (36 correct refusals), so the accuracy half is what makes the system useful. TrustSQL and EHRSQL report the same baseline ("abstain all": 50 and 20).

**Sensitivity to the penalty,** recomputed from `graded.csv` on 24 Oct. The decision used RS(10), fixed before the run.

| Setup | RS(0) | RS(5) | RS(10) | Break-even penalty | Accuracy |
|---|---|---|---|---|---|
| Our 7B, 4-bit | 72.1 | −8.8 | −89.6 | 4.46 | 69.0% |
| Our 7B, full precision | 76.1 | +3.7 | −68.7 | 5.26 | 73.6% |
| VL-3B, 4-bit | 45.1 | −160.3 | −365.7 | 1.10 | 46.4% |
| VL-3B, full precision | 55.9 | −99.0 | −253.9 | 1.80 | 56.3% |
| Coder-1.5B, 4-bit | 52.9 | −115.5 | −283.8 | 1.57 | 56.7% |
| Coder-1.5B, full precision | 52.5 | −124.2 | −301.0 | 1.49 | 55.6% |
| Coder-3B, 4-bit | 73.1 | −44.8 | −162.6 | 3.10 | 72.0% |
| Coder-3B, full precision | 74.1 | −38.7 | −151.5 | 3.28 | 73.9% |

- The break-even penalty is the penalty at which a model's score reaches 0. Our 7B is net useful only if one wrong number costs no more than about 4–5 right answers.
- At a penalty of 5, only our 7B in full precision is above 0. The decision is the same at any penalty, because no model reached the 90% half.

**Other weightings considered, and why they weren't used:**
- **Each question type counting equally** (a macro average): our 7B in 4-bit goes from 69.0% to 66.3%. Not used, because each item is a real question a responder could ask; shown here for transparency.
- **By importance,** for example a destroyed-building count weighing more than the smallest intact building's area: needs weights agreed with responders before a run. Usable in the re-test if agreed in advance.
- **By the size of the error** (off by 1 versus off by 100): a query with wrong logic can be close on one table and far off on another, and a responder can't see how far off a number is.
- **RS(N),** where one error outweighs all correct answers (EHRSQL): no system in either paper passes it except one that refuses almost everything, so it can't separate models.

**For the re-test,** set before the run: test-suite accuracy, RS at 0, 5 and 10, a risk–coverage curve if the system has an abstention gate (accuracy as the gate lets fewer questions through, the selective-prediction view both papers cite), and any importance weights agreed in advance.
- Summarise the risk–coverage curve by its RC-AUC (the area under risk against coverage; lower is better), and report the confidence's expected calibration error. Both are defined in TimeSpot (ICML 2026, Appendix D.1; Paper_Audit §4.15), which is the source to cite.

### 3.7 Free-form questions in the remote-sensing VLM papers (checked 24 Oct)

These projects do take free-form or conversational questions, but none tests those answers the way H2b did:

| Project | How it answers | How it was scored | Penalty for wrong answers, or refusals |
|---|---|---|---|
| TEOChat | The VLM answers from the image sequence, over several turns | Closed tasks: building localization and damage classification (F1), change detection | No |
| BDAChat (repository) | Segmentation (a modified SAM) finds buildings, then the VLM (Video-LLaVA, 7B) grades and explains each one | Damage classification: macro F1 0.763 on xBD | No |
| GeoPixel | The VLM writes a description with a mask for each object it mentions | Caption metrics plus mask overlap; referring segmentation | No |
| ImageRAG | Finds the relevant crops of a very large image, then the VLM answers | Multiple-choice accuracy (MME-RealWorld-RS, including 1,255 counting questions) | No |
| Change-Agent | An LLM writes Python that calls a change-detection model, for example to count changed buildings | Only the detection and captioning model is scored (IoU, BLEU, CIDEr). The agent's answers are shown in two example conversations, never measured | No |
| Grace | A small VLM on the satellite, a larger one on the ground | VQA and caption accuracy, and latency | No |
| DisasterInsight | The VLM answers building-level tasks, such as "Count the commercial buildings with major-damage" | Counting by mean absolute error; reports by BLEU, ROUGE and BERTScore | No |
| DisasterM3 | The VLM; its nine tasks include an open caption and restoration advice | Multiple-choice accuracy for six tasks, including building counting; open tasks scored by GPT-4.1 on a 5-point scale | No |

- **They ask a different question.** These papers ask whether a model can produce a good answer. H2b asked whether a system can ship free-form numeric answers to responders. No paper checks free-form numbers strictly, or scores a confident wrong answer below "I can't answer".
- **Change-Agent is the closest design to our SQL agent** (an LLM writing code to compute the answer), and its answers were only demonstrated, not measured.
- **BDAChat's design is closest to our current system:** find each building first, then let the VLM grade it, as our segmenter and table do.
- **Where they do count, it's weak.** In DisasterM3 the base 7B counts damaged buildings at 34.2%, and our 7B at about 30–32%. That's why counts come from the table.

**What H2b did and didn't rule out.** H2b tested free-form questions turned into database queries, which produce numbers. It didn't test descriptive questions answered by the VLM, such as "what kind of disaster is this?" or "describe the damage near the river" (our 7B gets disaster type 75.6% right at full size, L0 step 1). A descriptive question box could be offered, labelled as an AI description, with numbers still routed to fixed queries. It would need its own check before shipping, for example DisasterM3's 5-point judge scoring or a human rubric fixed in advance.

## 4. The showcase: what the public sees (T3)

**Status:** agreed 24 Oct. There are two pieces, and the demo (piece 2) is an interactive, animated website deployed on Vercel. Abrar builds it and chooses the name and look later. The build steps are in the roadmap's T3 section.

### 4.1 What last season's winner showed (observed 24 Oct)

DhakaDriveX was Champion at NSU ACM SC's Innovation Challenge (Season 19, 84 projects). Its "Project UI & Demonstration" link opens a static site on GitHub Pages (`nipunsaif.github.io/CSE499B`).

- **No live model.** There's no upload, no inference and no backend: plain HTML, CSS and a little JavaScript.
- **One strong look:** a full-screen neon city illustration, a neon logo and a car crossing the bottom of the page, with the same palette and font on every page.
- **Results in motion:** four YouTube videos in pop-up players. Two are the team's own (pedestrian crossing intent, and BanglaTesla trajectory prediction). The other two come from another lab's channel (INVETT-UAH).
- **One interactive proof of the main claim:** a knowledge graph of about 500 nodes, linking pedestrians to features such as crossing, near junction and occluded. Visitors can drag and zoom it. It's built with vis-network.
- **A team page** with photos and contacts, and a menu that works on phones.

The lesson: the public piece needs a clear look, results in motion, and one interactive element that proves the main claim. The public doesn't need a live model, but a reliable live run is stronger evidence at the booth.

### 4.2 Our two pieces

1. **Public showcase site (the story):**
   - a name and a look taken from our subject (satellite images, damage colours), chosen by the team, not a borrowed theme;
   - before/after sliders on the gallery images, with the damage map on top;
   - screen recordings of the live offline run, all our own footage;
   - our version of their knowledge graph: a building map where clicking a building shows its class, its area and its distance to the nearest destroyed building, from H2's table, so every number traces back to a building;
   - the benchmark panel with our real numbers, a limitations section, and the team.
2. **Interactive demo website on Vercel (the product):**
   - the chain in roadmap T3: a gallery image or an upload, the damage map, counts linked to their buildings, the heatmap, fixed question templates, the report and the benchmark panel;
   - animated, with current UI/UX (§4.4).

Both can live in one Next.js project, with a story page and a demo page: one design, one codebase, one deployment.

### 4.3 How the website runs our models

- **Vercel can't run them.** It has no GPU, and its serverless functions are far too small for PyTorch and a 7B. So the public deployment works from results computed in advance for the gallery images.
- **Live mode at the booth, offline:**
  - The same front end, built on the laptop, calls a local Python API (for example FastAPI). The API runs the chain: the segmenter, H2's fixed queries, T1's heatmap and T2's report, streamed as it's written.
  - The 7B needs a GPU with about 6 GB of memory in 4-bit (H2b measured 5.8 GB). Without one, the gallery's reports computed in advance are shown, labelled as such.
- **Stretch: uploads on the public site with no server (decided 24 Oct: try it).**
  - ONNX Runtime Web can run an ONNX model in the visitor's browser, so their images never leave their device.
  - Only the segmenter, counts and heatmap could run this way, not the report.
  - **Feasibility test first**, with its pass rule fixed before it runs:
    - Model: L1b's INT8 edge model, `Champion_Task/L1B-results/L1b/r9_int8_static_keep1_3.onnx` (33 MB). It is R9, which sees only the after image. R11's version needs L1 again, but its speed should be about the same, since only the first layer differs.
    - Run it in Chrome and Edge on the demo laptop with ONNX Runtime Web (WebAssembly, several threads), and the full-precision model (`L1-results/L1/r9_fp32.onnx`, 130 MB) on WebGPU where the browser offers it.
    - Measure: whether it loads (every operation supported), the time per 1024 × 1024 image (first and later runs), and agreement with the same model run natively on the same input. On a normal CPU, the native INT8 model takes 1.6 s.
    - Pass: it runs in both browsers, takes 10 s or less per image after the first, and gives the same class as the native run on at least 99.9% of pixels.
    - Note for the build: several threads in the browser need cross-origin isolation, so the Vercel deployment must send the COOP and COEP headers.
  - **Result (24 Oct): passed, in two runs.** Run on this PC (20 threads, with a GPU) in headless Chrome 153 and Edge 153, with ONNX Runtime Web 1.20.1, on one xBD after-image (Hurricane Harvey) at 1024 × 1024. The second run used a fresh copy of the test kit, to check the kit itself:

    | Model and engine | Chrome: first run / later runs | Edge: first run / later runs | Same class as the native run |
    |---|---|---|---|
    | INT8 (33 MB), WebAssembly | 3.3–4.0 s / 2.1–4.8 s | 4.8–6.4 s / 3.3–5.5 s | 100% of pixels, both runs |
    | Full precision (130 MB), WebAssembly | 3.5–5.0 s / 3.9–6.5 s | 4.6–5.9 s / 3.7–4.3 s | 100% |
    | Full precision, WebGPU | 4.8–5.6 s / 1.2–1.7 s | 3.4–3.8 s / 1.0–1.3 s | 100% |
    | INT8, WebGPU | failed | failed | — |

    - Timings vary between runs on the same PC, so read them as ranges. Agreement and the per-class pixel counts were identical in both runs, and every later run stayed under the 10 s limit.
    - The same models run natively (onnxruntime-node 1.20.1, CPU) take 0.75–1.05 s, so the browser takes about 2.5–7 times as long on WebAssembly.
    - INT8 on WebGPU fails in this version: its DequantizeLinear kernel doesn't support int32 values without a zero point. Options: a newer ONNX Runtime Web, re-quantizing with zero points, or a half-precision model (about 65 MB) for WebGPU.
    - **Recommended setup:** INT8 on WebAssembly as the default (it works in both browsers and is a 33 MB download), and a full- or half-precision model on WebGPU as the faster option where the visitor's browser offers it.
    - **How it was scored, and why:** agreement with the native run on every pixel, because the question was whether the browser runs the same model faithfully; and the time per image after the first run, because that's what a visitor waits. Accuracy against ground truth wasn't the question: the model is the same, and its accuracy was measured in L1b.
    - **Alternatives:** IoU against the native mask gives the same answer when agreement is 100%. A comparison on all 645 validation images is left for R11's version. A visual check is subjective.
    - **Caveats:** one image, on a fast PC, in headless browsers. A typical laptop has fewer threads and phones are slower, so re-test on the booth laptop and a mid-range phone. The test model is R9 (after image only); R11's version needs L1 again. Without the COOP and COEP headers, the browser runs on one thread.
    - The test kit is in [Champion_Task/T3-browser-test/](Champion_Task/T3-browser-test/README.md): the scripts, the pinned packages, this run's results, and a README with the caveats and an audit checklist for teammates or their AI helpers. The models and generated files stay out of git.
- **Gradio** (`support/Abrar/Implementation_DisasterM3/gradio_app.py`) stays as the API's test page, and the fallback if the website isn't ready. Its `share=True` opens a public tunnel, so it must be off for offline runs.

### 4.4 What "current UI/UX" means here, as checks

- **Interactive:**
  - a before/after slider, and damage layers switched on and off by class;
  - hovering over or clicking a building shows its facts, and each count links to its buildings;
  - a heatmap layer, the report appearing as it's written, and charts for the benchmark panel.
- **Animation with a purpose:** the steps of the chain appear in order, so the animation explains the pipeline. It respects the visitor's reduced-motion setting.
- **Works for everyone:** on phones and by keyboard, with readable contrast, and clear loading, empty and error states.
- **Fast:** compressed images and lazy loading, with a measurable target such as Lighthouse 90+ for performance and accessibility.
- **Ready for the booth offline:** fonts and assets are bundled with the build (Next.js can host fonts itself), and live mode makes no outside calls.
- **Honest:** every number is labelled as a model estimate. Gallery images come from the validation set, so the site can show the real accuracy on each one.

### 4.5 What has to exist first

- **T2's reports:** DisasTeller on local models.
- **A gallery export:** a short Kaggle notebook that runs R11 with TTA on 4–6 validation pairs and saves the masks, the building outlines with their facts, the heatmaps and each image's accuracy.
- **L1 repeated on R11,** if the CPU and in-browser segmenter must match the champion. L1b's INT8 model is R9, which sees only the after image.
- **A Vercel account** linked to the repository. The free Hobby plan covers non-commercial projects.
- **Abrar's choice of name and look,** to be made later.

## 5. Beyond Layer 1: the day-1 Layers 2–5

The day-1 proposal stacked five layers and marked only Layer 1 as this project:

| Layer | What it is | Day-1 label |
|---|---|---|
| 5 | Unified 3D dashboard (UI/UX) | — |
| 4 | Inverse physics, collapse trajectory (PINNs) | research frontier |
| 3 | Generative pre-disaster reconstruction (diffusion) | research frontier |
| 2 | 3D reconstruction (NeRF, Gaussian splatting) | active research |
| 1 | VLM aerial image to textual rescue guidance | this project |

Layer 1 grew well beyond the day-1 arrow (image, VLM, text). The VLM alone couldn't read damage reliably (our 7B counts buildings at about 30–32%), so Layer 1 now includes:
- the damage segmenter (O1 met on all 645 validation images. On the 605 left after E5 removed 40 leaked pairs, only R16c step 5's SAM 3-only map passes: 0.3161 and 0.3051 Destroyed IoU, narrowly. A third grader seed, step 5b, is built and ready to run (27 Oct). SAM 3 + R11's pieces scores 0.2838–0.2980, 26 Oct);
- per-building counts from a table (H2), and the H2b test of free-form questions;
- the heatmap (T1) and the edge segmenter (L1b);
- the stronger 7B and the 3B path (L0, L2).

The T3 website (map, counts, heatmap and report on one screen) is a 2D version of Layer 5's dashboard.

### 5.1 The smallest version of each layer that would stand (estimates, 24 Oct)

These are rough estimates for one person working mainly on that layer.

| Layer | Smallest version | What it shows | Time | GPU | Risk |
|---|---|---|---|---|---|
| 2 or 5 | **3D damage city** in the website: building outlines raised into 3D blocks, coloured by damage, with schematic heights | A 3D view of our 2D results; strictly a Layer 5 piece, not reconstruction | about 1 week | none | low |
| 2 | A real 3D reconstruction (Gaussian splatting) of one disaster site, from openly licensed drone video, viewable in the browser | That the technique works; not connected to our satellite pipeline, since 3D needs images of one place from several angles, which our data lacks | 2–3 weeks | 5–10 h | medium: licensed footage, and camera positions can fail on blurry video |
| 3 | A model that turns an after-image into the likely before-image, trained on our 5,798 training pairs: a pix2pix baseline, then a diffusion model | Tested by running R11 with the real before-image, a generated one, and none (R9) on the 645 validation images. If generated beats none, it helps when no real before-image exists. Either outcome is a result | 3–4 weeks | 30–50 h | medium: slow diffusion training on a T4, and invented buildings |
| 4 | For one earthquake, fit a physics-based model (damage falls off with distance from the source) to the per-building damage grades, and compare the inferred shaking with the USGS ShakeMap | An inverse-physics result checked against real data; a PINN is optional | 3–4 weeks | little | high: needs map coordinates for each image (not yet checked), and earthquakes are our weakest case (E4) |

Real collapse trajectories (how each building fell) need structural data and simulations: months of work, out of reach this semester. All three of Layers 2–4 together would take about 7–9 person-weeks, competing with T2, T3, L0 and L2, which O3 and O4 need. None of Layers 2–4 is in O1–O4.

### 5.2 Decisions

- **Agreed 24 Oct:** T2, T3, L0 and L2 come first.
- **Agreed 24 Oct:** the 3D damage city goes into the T3 website, labelled as a 3D view of our 2D results.
- **Open:** Layer 3, if time allows after the above; recommended as the one extra layer, because it uses data we already have, plugs into R11 and gives a measured answer either way.
  - The final showcase is about 15 Dec (tentative, given 24 Oct), about 7 weeks away. T2, T3 (which Abrar builds), L0, L2 and H2c take most of that.
  - Layer 3 fits only as a side track: notebooks built for teammates to run on their GPU accounts, with exact steps. Decide in mid-November, from where L2 and the website stand.
- **Recommended, not yet decided:** Layer 4 and real 3D reconstruction stay as future work.
