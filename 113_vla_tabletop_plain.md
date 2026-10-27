# 113 in plain words

**What:** teach a robot arm (in simulation) to follow typed instructions like "put the red cube on the green pad",
by fine-tuning a small open vision-language-action model (SmolVLA) on demonstrations from a scripted expert.

**Why:** to learn how these models work, and to see whether the language part really matters.

**Status (27 Oct 2026): the project has ended.** Two things passed: picking up a single cube (96.8 %, after the owner
made the task easier: the cube spawns within ±15° of straight), and picking up the cube named by colour among 2–4
cubes (96.7 %). Sizes, placing on a pad and stacking did not pass. The final comparison stage, the larger model
(pi0.5) and the no-language comparison network never gave a usable result: the first two never ran, and the third
failed even on its own training scenes.

**The story.**
- **The arm learned to grab and lift a cube** from camera images alone.
- **At first it ignored the words.** With several cubes it picked one about as if the instruction were not there; a
  shuffled instruction worked as well as the right one.
- **Why (25 Oct check):** the colour word did reach the part of the model that plans the motion, but the plan hardly
  changed with the word; our reading is that the model's own random starting noise decided where the arm went. Pushing harder on the word
  at run time, up to 8 times, did not help (29.7 % at best vs 28.7 % without).
- **The fix was in training.** A second loss term made the model pay when the same motion fits another cube's
  instruction as well as the right one. After that it went for the named cube in 95.5 % of its grasps (chance is
  34.5 %). The grasp itself got less precise at first; more demonstrations and a longer, gentler training run
  brought it back: 96.7 % with colours.
- **Sizes ("the small red cube", up to 5 cubes) failed:** the model understood the size words (95.0 % of grasps on the
  named cube), but in crowded scenes the arm sometimes swung into the table or closed beside the cube: 79.4 % at best,
  bar 85 %.
- **Placing on a pad failed (72.7 %, bar 80 %), stacking failed (40.7 % on validation, bar 70 %).** For placing, a check showed the
  model went for the right cube and the right pad every time; the misses were in gripping precisely and letting go: off-centre, too late, or dropping the cube on
  the way. Learning to place also made its picking a bit worse (90.2 %, from 96.7 %).
- **One extra round of training for placing made it worse** (75.0 % down to 64.0 % on the same practice scenes): our reading
  (not tested) is that restarting training on a finished model pushed it away from its best point.

**Final check (27 Oct, reported only, no pass/fail):** on the scenes they were trained on, the models scored about as in
their tests. With a different table colour and lighting, every model fell apart (18.7 % at best); none was trained on
such scenes (out of distribution for all of them). With cube-and-size combinations never used as targets in training, the colours model mostly grabbed a
different cube it had been trained to pick (17.7 % success); our reading is that it learned "which cube" rather than combining colour and
size; with reworded instructions the single-cube and colours models kept working, but the placing model, asked in
other words to pick up a cube, mostly picked it up and put it back down (12.2 % success).

**What it shows:** a small VLA fine-tuned on a few thousand demonstrations learns the arm motion from pixels; it only
uses the words once the training loss forces it to, and after that the remaining failures are about motor precision,
not language.

**See it:** a 5-minute video (`media/113_demo.mp4`) shows, for each skill, what the robot was told, how success
was judged, and the first 16 episodes on new scenes, unselected, with the simulator's verdict on each; the figures in
`figures/` explain the method. The code is public; the model weights and datasets are not.

**Cost:** 178.95 of 200 H200 GPU-hours (+0.40 for the demo video) on PERUN (the owner raised the cap from 100 in steps
to 200), 8.07 of 20 laptop GPU-hours.
