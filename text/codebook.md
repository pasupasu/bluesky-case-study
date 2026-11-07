# Stance Classification Codebook

This codebook sets the rules for labeling Bluesky posts about the Israel–Palestine conflict with one of **seven stance labels**. Each post gets exactly one label, a confidence score and optional notes. The label values below match the dropdown in the annotation sheets (`STANCES` in `src/bluesky_case_study/annotation.py`).

| Label | Short definition |
|---|---|
| `pro_palestinian` | Sides with Palestinians or condemns Israel as the aggressor |
| `pro_israeli` | Sides with Israel or condemns Hamas/Palestinians as the aggressor |
| `anti_war_humanitarian` | Focuses on victims and ending the violence without blaming a side |
| `critical_of_both` | Explicitly criticizes both Israel and Hamas/Palestinian actors |
| `neutral_news` | Reports events factually with no evaluative framing |
| `unclear` | About the conflict, but no stance can be recovered |
| `unrelated` | Not about the Israel–Palestine conflict |

---

## 1. Unit of analysis and general rules

- **Code the post text as it appears in the sheet.** Don't look up the author, their other posts or the thread. Author information is hidden on purpose so labels reflect the post, not the person.
- **Code the stance toward the conflict, not the tone.** An angry post and a calm post can share a label.
- **Code what the post communicates, not only its literal words.** Read sarcasm, irony and rhetorical questions for their intended meaning, e.g. *"Sure, bombing refugee camps is self-defense"* is `pro_palestinian`. If you can't tell whether a post is sarcastic, use `unclear` or lower your confidence.
- **Links and quoted material.** If the post only shares a link or quote with no added comment, code the stance of the shared material as far as you can tell from the text. If the poster adds a comment, the comment wins.
- **One label per post.** If a post seems to fit two labels, use the decision procedure (Section 3) and the boundary rules (Section 4).
- **Don't infer stance from keywords alone.** Words like *Gaza*, *IDF* or *Hamas* show the topic, not the side. Look for evaluation: who is blamed, defended, mourned or praised.

---

## 2. Category definitions

### `pro_palestinian`

The post sides with Palestinians or presents Israel as the main wrongdoer.

**Include when the post does any of the following:**
- Calls Israel's actions genocide, ethnic cleansing, apartheid or occupation, or calls Israel a colonial or settler state.
- Expresses solidarity with Palestinians or Palestine as a people or state (e.g. *"Free Palestine"*, *"From the river to the sea"*, 🇵🇸 used as a statement).
- Calls Israelis "settlers" or "colonists" as a general description.
- Blames Israel, the IDF or Israeli leaders for civilian deaths, or calls for sanctions, boycotts, divestment or an arms embargo against Israel.
- Says that accusations of antisemitism are being used to shield Israel from criticism.
- Frames Palestinian armed resistance as legitimate.

**Exclude:**
- Calls for a ceasefire or grief for Gazan civilians that blame no one → `anti_war_humanitarian`.
- Criticism of Israel together with explicit criticism of Hamas → `critical_of_both`.

### `pro_israeli`

The post sides with Israel or presents Hamas, Palestinian groups or their supporters as the main wrongdoer.

**Include when the post does any of the following:**
- Defends Israel's military actions or its right to self-defense.
- Condemns Hamas, Hezbollah, Iran or "Palestine" as the aggressor, without also criticizing Israel.
- Focuses on Israeli victims or hostages in a way that justifies Israel's response or blames Palestinians.
- Answers criticism of Israel by calling it antisemitic, or treats pro-Palestinian protest as inherently antisemitic or terrorist-supporting.
- Denies or minimizes Palestinian civilian deaths (e.g. *"the Gaza death toll is Hamas propaganda"*).

**Exclude:**
- General condemnation of antisemitism with no link to the conflict → `unrelated`.
- Concern about antisemitism linked to the conflict that doesn't defend Israel or attack its critics → `unclear`.
- Grief for Israeli victims or calls to free hostages with no blame or justification → `anti_war_humanitarian`.

### `anti_war_humanitarian`

The post focuses on the victims of the conflict and on ending the violence, without blaming either side.

**Include when the post does any of the following:**
- Calls for a ceasefire, peace, de-escalation or an end to the killing without naming a culprit.
- Mourns civilians (Palestinian, Israeli or both) without assigning blame.
- Calls for aid, hostage release or protection of children, hospitals or journalists without blaming a side.
- Expresses general pacifism or opposition to war and genocide, applied to this conflict, without directly condemning Israel or Hamas.

**Exclude:**
- Ceasefire calls aimed at Israel or framed around Israeli guilt (e.g. *"Israel must stop the genocide now"*) → `pro_palestinian`.
- Hostage-focused posts that justify the war → `pro_israeli`.
- Factual reports of casualties or aid shortfalls with no call to action or emotional framing → `neutral_news`.

### `critical_of_both`

The post **explicitly** criticizes both sides: Israel or its government on one hand, and Hamas or other Palestinian armed groups on the other.

**Include:**
- *"Netanyahu and Hamas are both war criminals."*
- Posts condemning both the October 7 attack and Israel's conduct in Gaza.

**Exclude:**
- Posts that criticize one side and merely mention the other.
- Even-handed grief with no criticism of either side → `anti_war_humanitarian`.
- Posts that criticize both sides' online supporters but not the actors themselves → `unclear` (unless a stance on the conflict is recoverable).

### `neutral_news`

The post reports facts or events with no evaluative language.

**Include:**
- Headlines and event summaries (e.g. *"Talks on hostage deal resume in Cairo"*).
- Links to mainstream wire or news outlets (Reuters, AP, BBC, etc.) shared without comment, or with a purely descriptive comment.

**Exclude:**
- Reports focused on civilian suffering (casualty counts framed around children, famine, destroyed hospitals) shared in an emotive or advocacy frame → `anti_war_humanitarian`, or a partisan label if blame is assigned.
- Headlines with loaded wording (*"massacre"*, *"genocide"*, *"terrorists"*, *"occupation forces"*) → code the stance that wording signals.
- Satire, opinion pieces and praise or endorsement of an analysis ("great thread", "must-read") → code the stance of the content being endorsed.
- **Partisan outlets.** The source matters. Content from outlets with a clear editorial line on the conflict (e.g. Iranian state media such as Press TV on one side, or explicitly advocacy-oriented outlets on the other) takes that outlet's stance, even if the headline reads neutrally.

### `unclear`

The post is about the conflict or the discourse around it, but you can't recover a stance on the conflict itself.

**Include:**
- Too short or vague to read (e.g. *"Thoughts on Gaza?"*, a lone flag emoji with no context).
- Meta-discussion about the debate (moderation, blocking, arguments online) that doesn't reveal a side.
- Possible sarcasm you can't resolve.
- Replies that only make sense with the parent post (e.g. *"Exactly this."*, *"Source?"*).

**Use `unclear` only after ruling out the other labels.** Don't use it for posts that are hard to call between two stances. Pick the best fit and lower your confidence instead.

### `unrelated`

The post isn't about the Israel–Palestine conflict.

**Include:**
- Keyword false positives (e.g. "Hamas" in an unrelated sense, a *"Gaza Strip"* bar name, or "settlers" about US history).
- Other conflicts or genocides (Sudan, Ukraine, Congo) with no link to Israel–Palestine.
- Antisemitism or Islamophobia discussed with no link to the conflict.
- Non-English posts. Don't translate them; code them `unrelated` and write the language in `notes`.

**Exclude:**
- Posts that compare another conflict to Gaza or Israel → code the stance toward Israel–Palestine that the comparison implies.

---

## 3. Decision procedure

Work through these questions in order and stop at the first that applies.

1. **Is the post in English and about the Israel–Palestine conflict, its actors or the discourse around it?** No → `unrelated`.
2. **Does the post explicitly criticize both Israel and Hamas/Palestinian armed groups?** Yes → `critical_of_both`.
3. **Does the post side with, defend or praise one side, or blame, condemn or delegitimize the other?** Yes → `pro_palestinian` or `pro_israeli`.
4. **Does the post focus on victims, peace or ending the violence without blaming a side?** Yes → `anti_war_humanitarian`.
5. **Does the post report facts without evaluative language or advocacy framing?** Yes → `neutral_news`.
6. Otherwise → `unclear`.

---

## 4. Boundary rules

| If the post… | Label |
|---|---|
| Calls for a ceasefire and names Israel as responsible | `pro_palestinian` |
| Calls for a ceasefire and names no one | `anti_war_humanitarian` |
| Condemns Hamas only | `pro_israeli` |
| Condemns Hamas and Israel | `critical_of_both` |
| Mourns Israeli hostages, no blame or justification | `anti_war_humanitarian` |
| Mourns Israeli hostages and justifies the war | `pro_israeli` |
| Calls criticism of Israel antisemitic | `pro_israeli` |
| Says antisemitism accusations are used to shield Israel | `pro_palestinian` |
| Condemns antisemitism, no link to the conflict | `unrelated` |
| Shares a Reuters/AP headline without comment | `neutral_news` |
| Shares a partisan outlet's headline without comment | that outlet's stance |
| Praises or endorses someone else's analysis | the stance of that analysis |
| Uses "genocide" about Gaza | `pro_palestinian` |
| Uses "genocide" in general or about another conflict | `anti_war_humanitarian` if applied to this conflict without blame, else `unrelated` |

---

## 5. Confidence

Every post gets a confidence score in the `confidence` column:

| Score | Meaning |
|---|---|
| **3** — certain | The label is clear from the text; another coder would almost surely agree. |
| **2** — fairly sure | The label is the best fit, but the post is short, ambiguous or relies on implied meaning. |
| **1** — unsure | You chose between two or more plausible labels. Write the alternative in `notes`. |

Low-confidence posts are where the codebook most needs work, so write in `notes` why a post was hard.

---

## 6. Notes column

Use `notes` for:
- The runner-up label when confidence is 1 (e.g. `alt: anti_war_humanitarian`).
- The language of non-English posts.
- Suspected sarcasm, missing context or reply-only posts.
- Cases the codebook doesn't cover, so the rules can be updated.

---

## 7. Revising the codebook

Revise this codebook using the **dev** set only. Never read or discuss **test** posts while changing the rules. When you change a rule, record the date and what changed below, and recheck any dev labels it affects.

| Date | Change |
|---|---|
| | |
