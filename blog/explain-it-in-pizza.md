# The Human Baseline Problem — Pizza Version

## The Experiment, Explained With Pizza

---

### What We Were Trying to Do

Imagine you work at a pizza quality control lab. Your job is to look at a pizza
and answer one question: **was this made by a human or by a robot pizza machine?**

You figure: robot pizzas probably have consistent sauce coverage, perfectly even
cheese distribution, identical crust thickness every time. Human pizzas have
character — a little uneven, sometimes more cheese on one side, the crust a bit
thicker where the pizzaiolo was distracted.

So you train a classifier. You collect 500,000 pizzas — half human-made, half
robot-made — and you teach a model the difference. It gets 95% accuracy. Great!

Except.

---

### The Three Pizza Shops

For the training data, your "human" pizzas came from **one specific pizza shop**:
a busy New York street-corner slice joint. Fast, messy, efficient. Thin crust,
irregular sauce, rushed.

Then you test your detector on pizzas from **two other places**:

- **Shop 2**: A Codeforces Pizza (competitive pizza-making competition). These human
  pizzas are made by people trying to win. They're meticulously constructed,
  perfectly sauced, evenly topped — a competitive pizzamaker's best work.

- **Shop 3**: HumanEval Pizza (OpenAI's pizza recipe test). These are the canonical
  reference pizzas written by expert chefs for a textbook. Perfect technique,
  perfect ratio, exactly what a "correct" pizza should look like.

Your detector had learned: **messy = human, clean = robot**.

So when you run it on the competition pizzas and the cookbook pizzas — which are
clean, precise, and beautifully made — **it calls them all robot pizzas**.

Your accuracy drops to 33–40%. Worse than guessing.

---

### Why This Happens (The Real Finding)

Here's the key insight: **the robot machines are consistent everywhere**.

Robot pizza at Shop 1, Shop 2, and Shop 3? Basically the same. The robot does what
the robot does. Consistent sauce pressure, consistent timing, consistent output.

**But humans are all over the place.** The rushed New York guy makes a very different
pizza than the competitive champion, who makes a very different pizza than the
cookbook expert. Humans vary enormously based on who they are and what they're trying to do.

Your detector didn't learn "what robot pizza looks like." It learned "what the
**New York street corner human** doesn't look like." Those are completely different
things.

When you show it a cookbook pizza — clean, precise, beautifully made — it says ROBOT.
Because it never met a human who cooks that well.

---

### The Features That Lie

We measured exactly which pizza features caused the problem:

| Feature | NY Slice (human) | Robot | Cookbook Expert (human) |
|---|---|---|---|
| Sauce evenness | 0.03 | **0.16** | **0.15** |
| Cheese coverage | 0.26 | **0.45** | **0.63** |
| Crust consistency | messy | clean | **clean** |

The cookbook expert human looks more like the robot than like the NY slice human
**on every single feature we measured**. 10 out of 10.

The detector had no choice. It called the cookbook expert a robot. Because in its
world, "human" means "NY slice joint."

---

### The Features That Don't Lie

Not all features flipped. Some were consistent across all three shops:

- How long the average line of sauce is (consistent direction everywhere)
- How much of the pizza uses snake-pattern toppings (consistent everywhere)
- Whether the crust tapers at the edges (consistent everywhere)

These 9 "stable features" are the ones that measure something real about robot
output — not just something real about the *contrast* between robots and one
specific type of human.

When we use only those 9 features, our cross-shop accuracy goes from 38% to 62%.
Not great, but almost twice as good. That's the signal that actually generalizes.

---

### The Fix

What if we just trained on pizzas from all three shops?

We mixed:
- 10,000 NY street corner pizzas (human + robot)
- 286 competition pizzas (human + robot)
- 328 cookbook pizzas (human + robot)

Accuracy: **95%**. Same features. Same model. Three times the human diversity.

The features were never wrong. We just only knew one kind of human.

---

### The HBD Score (The Honest Version)

We can't reliably say "this pizza was made by a robot." But we can say:

**"This pizza is X standard deviations from the average pizza we've seen from humans."**

That's the Human Baseline Distance (HBD) score. It's not a verdict. It's:
"this pizza is unusual relative to the humans we know."

What that tells you depends on context:

- **Codeforces context (competition shop)**: Human pizzas score HBD ~1.8, robot pizzas
  score HBD ~8.0. AUROC = 0.975. Very useful. The competitive humans are very distinct
  from the robots.

- **SemEval context (NY street corner)**: Humans score ~1.8, robots ~4.6. AUROC = 0.798.
  Pretty useful.

- **HumanEval context (cookbook expert)**: Humans score ~4.35, robots ~4.56.
  AUROC = 0.540. **Useless.** The expert humans and the robots are making pizza
  at the same quality level. The scorer honestly tells you: "I can't tell these apart,
  because they're both better than the average pizza I've seen."

That last number — 0.540 — is not a bug. It's the whole point. The scorer is honest
about what it doesn't know.

---

### The Model Family Finding (Bonus Pizza Round)

Can you tell *which robot* made a pizza from the pizza itself?

We tried to classify 34 different robots + human from style alone. We got ~23% accuracy
where random chance is 3%. About 8x above chance.

Interesting stuff:
- The Phi-3 robots make pizzas that look almost identical to each other (same factory,
  different batch sizes)
- The StarCoder 15B and StarCoder 3B robots can barely be told apart (23% confusion —
  bigger robot, same recipe)
- **Human pizza is the most recognizable class** (F1 = 0.622) — within the NY shop
  data, human messiness is more distinctive than any robot's particular style

---

### What This All Means

You cannot reliably detect AI-generated code by looking at surface style alone —
unless you know exactly what kind of humans you're comparing against.

High accuracy on a single dataset means: "we learned to distinguish this specific
human population from this specific AI." It doesn't mean "we can detect AI."

The test is: **does it still work when you change who the humans are?**

Ours doesn't. Until you train on all three kinds of humans. Then it does.

The lesson for anyone building or using AI code detectors:

> Before you trust any accuracy number, ask: whose pizza were the "human" examples?
> If the answer is "one kind of human," the number is measuring that human's style,
> not AI authorship.

---

*This is a simplified explanation of the SentinalAI paper:*
*"The Human Baseline Problem: Why AI Code Detectors Fail Across Domains"*

*Want the actual numbers? [Read the full blog post with real data →](post)*  
*Want the academic paper? [PAPER.md](../PAPER.md)*
