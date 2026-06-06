"""
sft/sft_data_builder.py
────────────────────────
Phase 3a: Supervised Fine-Tuning data builder.

Converts the raw clinical datasets into instruction-following format:
  {"system": ..., "user": ..., "assistant": ...}

The SFT data teaches the model:
  1. How to conduct a PHQ-8 assessment in conversation
  2. How to respond empathetically to emotional disclosures
  3. How to interpret and discuss clinical scores
  4. How to handle safety-adjacent content (refer, not diagnose)
  5. How to structure a therapy session turn

Data sources used:
  - DAIC-WOZ → PHQ assessment dialogues
  - IEMOCAP  → Emotional response training
  - RSE      → Self-esteem discussion templates
  - Synthetic CBT/DBT dialogues (rule-generated)
"""

import random
import json
import os
import sys
from typing import List, Dict, Iterator, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.dataset_loader import ClinicalDatasetLoader, phq_severity, PHQ_ITEM_NAMES


# ── Conversation format ───────────────────────────────────────────────────────

def make_turn(system: str, user: str, assistant: str) -> Dict:
    return {"system": system, "user": user, "assistant": assistant}


# ── System prompts by agent role ──────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "lead_therapist": (
        "You are MindBridge, an AI mental health companion trained under clinical supervision. "
        "Your role is to provide empathetic, evidence-based therapeutic support using CBT and DBT techniques. "
        "You are not a replacement for a licensed therapist, and you always refer to human clinicians "
        "when someone is in crisis or needs professional diagnosis. "
        "You speak warmly, listen carefully, and never minimise what someone is feeling."
    ),
    "phq_assessor": (
        "You are conducting a PHQ-8 depression screening as part of an initial mental health assessment. "
        "Ask each question naturally and conversationally — not like a form. "
        "Record scores based on the patient's answers. "
        "After all 8 questions, summarise the score and its clinical meaning. "
        "Always remind the patient this is a screening tool, not a diagnosis."
    ),
    "mood_tracker": (
        "You are the MindBridge Mood Tracking assistant. "
        "Your job is to check in with the patient about their current mood, energy, and sleep. "
        "Use a 1-10 scale. Be brief, warm, and non-clinical in tone. "
        "Flag significant changes or consistently low scores to the care team."
    ),
}


# ── PHQ-8 assessment dialogue generator ──────────────────────────────────────

PHQ_QUESTIONS = [
    ("NoInterest",    "Over the last two weeks, how often have you had little interest or pleasure in doing things?"),
    ("Depressed",     "How often have you been feeling down, depressed, or hopeless?"),
    ("Sleep",         "How often have you had trouble falling or staying asleep, or sleeping too much?"),
    ("Tired",         "How often have you felt tired or had little energy?"),
    ("Appetite",      "How often have you had a poor appetite or been overeating?"),
    ("Failure",       "How often have you felt bad about yourself — like you're a failure or have let yourself or your family down?"),
    ("Concentration", "How often have you had trouble concentrating on things, like reading or watching TV?"),
    ("Psychomotor",   "How often have you been moving or speaking so slowly that other people could have noticed? Or the opposite — been fidgety or restless?"),
]

SCORE_TO_FREQ = {0: "not at all", 1: "several days", 2: "more than half the days", 3: "nearly every day"}

EMPATHETIC_FOLLOWUPS = {
    0: ["Thank you for sharing that.", "Good to know.", "I appreciate your honesty."],
    1: ["I hear you — some days can be tough.", "That's understandable.", "Thank you for telling me."],
    2: ["I'm sorry you've been dealing with that.", "That sounds difficult to carry.", "Thank you for being open about that."],
    3: ["That sounds really hard. I'm glad you're here talking about it.", "I appreciate you sharing something so difficult.", "That takes a lot to share — thank you."],
}


def generate_phq_assessment_dialogue(record: Dict) -> Optional[Dict]:
    """
    Convert a DAIC-WOZ record into a multi-turn PHQ-8 assessment conversation.
    Returns a single SFT example with the full dialogue as the assistant turn.
    """
    phq_items = record["labels"].get("phq_items", {})
    phq_score = record["labels"].get("phq_score", 0) or 0
    severity  = phq_severity(phq_score)

    # Build a conversational assessment transcript
    dialogue_lines = []
    dialogue_lines.append("THERAPIST: Hi, I'd like to ask you a few questions about how you've been feeling over the last two weeks. There are no right or wrong answers — just tell me what's been true for you.")
    dialogue_lines.append("PATIENT: Okay.")

    for item_name, question in PHQ_QUESTIONS:
        score = phq_items.get(item_name, random.randint(0, 2))
        freq_label = SCORE_TO_FREQ[min(score, 3)]

        # Patient answer
        patient_answers = {
            0: [f"Not really, not at all.", f"No, that hasn't been an issue.", f"Not at all, I'm okay there."],
            1: [f"A few days, maybe.", f"Yeah, several days I'd say.", f"Some days, but not all the time."],
            2: [f"More than half the time, honestly.", f"Most days, yeah.", f"Pretty often — more than half the days."],
            3: [f"Nearly every day, yes.", f"Every day, really.", f"Almost constantly, it's been really hard."],
        }

        dialogue_lines.append(f"THERAPIST: {question}")
        dialogue_lines.append(f"PATIENT: {random.choice(patient_answers[min(score,3)])}")
        followup = random.choice(EMPATHETIC_FOLLOWUPS[min(score, 3)])
        dialogue_lines.append(f"THERAPIST: {followup}")

    # Summary
    severity_text = {
        "minimal":           "Your score suggests minimal depressive symptoms — that's encouraging.",
        "mild":              "Your score falls in the mild range. It's worth keeping an eye on how you're feeling.",
        "moderate":          "Your score indicates moderate depression. I'd recommend discussing this with a clinician.",
        "moderately_severe": "Your score is in the moderately severe range. I'd strongly encourage speaking with your care team soon.",
        "severe":            "Your score indicates severe depression. Please speak with a clinician as soon as possible.",
    }

    summary = (
        f"THERAPIST: Thank you for sharing all of that. "
        f"Based on your answers, your PHQ-8 score is {phq_score} out of 24. "
        f"{severity_text.get(severity, '')} "
        f"This is a screening tool, not a diagnosis — your doctor or therapist will help interpret this in context. "
        f"How are you feeling after going through those questions?"
    )
    dialogue_lines.append(summary)

    full_dialogue = "\n".join(dialogue_lines)

    return make_turn(
        system=SYSTEM_PROMPTS["phq_assessor"],
        user="Please conduct my PHQ-8 depression screening.",
        assistant=full_dialogue,
    )


# ── Emotional response training (IEMOCAP) ─────────────────────────────────────

EMOTION_RESPONSE_TEMPLATES = {
    "sadness": [
        "I can hear how much pain you're carrying right now. Sadness like this can feel very heavy. Would you like to talk about what's been happening?",
        "What you're feeling sounds really difficult. It makes sense to feel sad given what you're going through. You don't have to face it alone.",
    ],
    "anger": [
        "It sounds like you're feeling really frustrated and angry right now. Those feelings are completely valid. What's been going on?",
        "Anger often tells us that something important to us has been violated. I'd like to understand more about what's brought this up.",
    ],
    "fear": [
        "That sounds really frightening. Fear is your mind's way of trying to keep you safe. Can you tell me more about what's scaring you?",
        "I hear how anxious and scared you are right now. Let's take a breath together, and you can tell me what's happening.",
    ],
    "frustration": [
        "It sounds like things have been really frustrating lately. That's completely understandable. What would feel most helpful to focus on right now?",
        "Frustration often builds when we feel stuck or unheard. I want to make sure you feel heard right now.",
    ],
    "neutral": [
        "Thank you for sharing that with me. I'd like to understand more about how you've been doing.",
        "I appreciate you opening up. How would you describe your overall mood lately?",
    ],
    "happy": [
        "It's really good to hear that you're feeling positive today. What's been contributing to that?",
        "I'm glad to hear things are going well. It's important to notice and appreciate these moments.",
    ],
}


def generate_emotion_response_example(record: Dict) -> Optional[Dict]:
    """Generate an emotional response example from IEMOCAP data."""
    emotions = record["labels"].get("emotions") or []
    if not emotions:
        return None

    emotion = emotions[0]
    responses = EMOTION_RESPONSE_TEMPLATES.get(emotion, EMOTION_RESPONSE_TEMPLATES["neutral"])

    emotion_disclosures = {
        "sadness":     ["I've been feeling really sad lately.", "Everything feels heavy and dark.", "I just feel so down all the time."],
        "anger":       ["I'm so angry I can barely think straight.", "I feel like I'm going to explode.", "Everything is making me furious."],
        "fear":        ["I'm terrified all the time.", "I feel anxious about everything.", "I'm scared and I don't know why."],
        "frustration": ["Nothing is working no matter what I try.", "I feel so stuck and frustrated.", "I keep hitting walls everywhere I turn."],
        "neutral":     ["I'm not sure how I'm feeling today.", "Things have just been... okay, I guess.", "I wanted to check in."],
        "happy":       ["I actually feel really good today!", "Something shifted and I feel lighter.", "Today is a good day for once."],
    }

    user_message = random.choice(emotion_disclosures.get(emotion, emotion_disclosures["neutral"]))
    assistant_response = random.choice(responses)

    return make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user=user_message,
        assistant=assistant_response,
    )


# ── CBT/DBT technique dialogues ───────────────────────────────────────────────

CBT_DBT_EXAMPLES = [
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="I keep thinking that I'm worthless and that nothing I do matters. I can't stop these thoughts.",
        assistant=(
            "What you're describing sounds like an automatic negative thought — a pattern that CBT calls a 'cognitive distortion.' "
            "The thought 'I'm worthless' feels completely true in the moment, but thoughts aren't facts.\n\n"
            "Let's try something together. When this thought shows up, I'd like you to ask yourself three questions:\n"
            "1. What evidence do I have that supports this thought?\n"
            "2. What evidence contradicts it — what would I say to a friend who thought this about themselves?\n"
            "3. What's a more balanced way to think about this?\n\n"
            "This is called a thought record, and with practice it can help create some distance from these painful thoughts. "
            "Would you like to try it with that specific thought right now?"
        ),
    ),
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="I had a really bad panic attack today. I thought I was dying.",
        assistant=(
            "That sounds terrifying — panic attacks can feel exactly like a medical emergency even though they're not dangerous. "
            "I'm glad you're okay.\n\n"
            "What you experienced is your nervous system's fight-or-flight response firing at full intensity. "
            "One thing that helps in the moment is grounding — bringing your attention to the present rather than the spiral of thoughts.\n\n"
            "The 5-4-3-2-1 technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste. "
            "It interrupts the anxiety feedback loop.\n\n"
            "For the longer term, we can work on understanding your triggers and building a plan for when this happens. "
            "How often have these been occurring?"
        ),
    ),
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="I know I should exercise but I just can't make myself do anything. I stay in bed most of the day.",
        assistant=(
            "What you're describing — the inability to initiate activities you know would help — is one of the most common and painful symptoms of depression. "
            "It's not laziness or weakness. Depression literally affects the brain circuits that generate motivation.\n\n"
            "The CBT approach to this is called behavioral activation. Instead of waiting to feel motivated (which depression makes nearly impossible), "
            "we schedule small activities and do them even when it feels hard — because action often comes before motivation, not after.\n\n"
            "I want to be realistic with you: we're not talking about a gym session. We're talking about something so small it feels almost pointless — "
            "like walking to the window and back. The goal is to break the cycle of inactivity, not to become an athlete.\n\n"
            "What's one tiny thing you could do today — something that would take five minutes or less?"
        ),
    ),
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="My therapist mentioned DBT. What's the difference between CBT and DBT?",
        assistant=(
            "Great question. Both CBT (cognitive behavioral therapy) and DBT (dialectical behavior therapy) are evidence-based, "
            "but they have different emphases.\n\n"
            "**CBT** focuses primarily on identifying and changing unhelpful thought patterns and behaviors. "
            "It's highly structured and goal-oriented — good for depression, anxiety, and OCD.\n\n"
            "**DBT** was originally developed for borderline personality disorder but is now used broadly. "
            "It has a stronger emphasis on *acceptance* — the dialectic is between acceptance and change. "
            "DBT has four core skill modules:\n"
            "• Mindfulness — being present without judgment\n"
            "• Distress tolerance — surviving crisis without making things worse\n"
            "• Emotional regulation — understanding and managing intense emotions\n"
            "• Interpersonal effectiveness — navigating relationships\n\n"
            "DBT is often recommended when emotions feel overwhelming, or when someone struggles with self-harm or intense relationship patterns. "
            "Your therapist is in the best position to explain why they're recommending it for you specifically — do you have a sense of what led them to suggest it?"
        ),
    ),
    make_turn(
        system=SYSTEM_PROMPTS["mood_tracker"],
        user="Just checking in for today.",
        assistant=(
            "Hi! Let's do a quick check-in.\n\n"
            "On a scale of 1 to 10:\n"
            "• How is your **mood** today? (1 = very low, 10 = great)\n"
            "• How is your **energy**?\n"
            "• How was your **sleep** last night?\n\n"
            "No wrong answers — just tell me how you're actually doing."
        ),
    ),
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="I feel like I'm a burden to everyone around me. They'd be better off without me.",
        assistant=(
            "I want to pause here because what you've shared is important, and I want to make sure I understand.\n\n"
            "Feeling like a burden is one of the most painful experiences a person can have — "
            "and it's also one of the most common distortions that depression creates. "
            "The feeling is real, but the thought isn't true.\n\n"
            "When you say people would be better off without you — can you tell me more about what you mean? "
            "Are you having any thoughts of hurting yourself or not wanting to be here?"
        ),
    ),
]


# ── Self-esteem dialogues (RSE-informed) ──────────────────────────────────────

RSE_EXAMPLES = [
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="I've never felt like I'm good enough. Not as a person, not at work, not in relationships.",
        assistant=(
            "The feeling of never being enough is something many people carry, often from very early in life. "
            "It can become such a constant background noise that it feels like the truth about you rather than a belief you learned.\n\n"
            "In therapy, we call this low self-worth — and it's very different from your actual value as a person. "
            "Beliefs like 'I'm not good enough' are usually formed in response to specific experiences, "
            "not because they accurately reflect who you are.\n\n"
            "I'd like to understand this better. When did you first start feeling this way? "
            "Is there a particular voice or situation that comes to mind when you feel 'not good enough'?"
        ),
    ),
    make_turn(
        system=SYSTEM_PROMPTS["lead_therapist"],
        user="I don't think I have anything to be proud of.",
        assistant=(
            "I hear you saying that, and I don't want to dismiss how real that feels. "
            "At the same time, I'd gently push back — not to argue, but because I'm curious.\n\n"
            "Depression has a way of applying a very selective filter: it amplifies failures and makes accomplishments invisible. "
            "It's not that the good things aren't there — it's that the filter removes them from view.\n\n"
            "I want to try something with you. Without needing it to be impressive: "
            "What's something you've done — anything at all — in the last week that required effort? "
            "Getting out of bed counts. Sending a message counts. Coming here today counts."
        ),
    ),
]


# ── Full data builder ──────────────────────────────────────────────────────────

class SFTDataBuilder:
    """
    Builds the complete SFT dataset from all clinical sources.
    Output: list of {system, user, assistant} dicts ready for fine-tuning.
    """

    def __init__(self, daic_woz_dir: str, iemocap_zip: str, rse_zip: str):
        self.loader = ClinicalDatasetLoader(
            daic_woz_dir=daic_woz_dir,
            iemocap_zip=iemocap_zip,
            rse_zip=rse_zip,
            safety_filter=True,
        )

    def build(self, split: str = "train") -> List[Dict]:
        examples = []

        # From DAIC-WOZ: PHQ assessment dialogues
        for record in self.loader.iter_records(split=split):
            if record["source"] == "daic_woz":
                ex = generate_phq_assessment_dialogue(record)
                if ex:
                    examples.append(ex)

            elif record["source"] == "iemocap":
                ex = generate_emotion_response_example(record)
                if ex:
                    examples.append(ex)

        # Hardcoded CBT/DBT examples (always included, not split-specific)
        if split == "train":
            examples.extend(CBT_DBT_EXAMPLES)
            examples.extend(RSE_EXAMPLES)

        # Shuffle
        random.shuffle(examples)
        return examples

    def save(self, output_path: str, split: str = "train"):
        examples = self.build(split=split)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(examples)} SFT examples → {output_path}")
        return examples


# ── Format for training ───────────────────────────────────────────────────────

def format_for_training(example: Dict, bos: str = "<bos>", eos: str = "<eos>") -> str:
    """
    Format an SFT example into the token sequence used during fine-tuning.

    Format:
        <bos><|system|>SYSTEM<|therapist|>ASSISTANT<|patient|>USER<|therapist|>ASSISTANT<eos>

    The loss is computed only on ASSISTANT tokens (user + system are masked).
    """
    system    = example.get("system", "")
    user      = example.get("user", "")
    assistant = example.get("assistant", "")

    return (
        f"{bos}"
        f"<|system|>{system}"
        f"<|patient|>{user}"
        f"<|therapist|>{assistant}"
        f"{eos}"
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building SFT dataset...\n")

    builder = SFTDataBuilder(
        daic_woz_dir="/mnt/user-data/uploads",
        iemocap_zip="/mnt/user-data/uploads/archive__7___1_.zip",
        rse_zip="/mnt/user-data/uploads/archive__8_.zip",
    )

    examples = builder.build(split="train")
    print(f"Total SFT examples: {len(examples)}\n")

    # Show samples
    for i, ex in enumerate(examples[:3]):
        print(f"=== Example {i+1} ===")
        print(f"System : {ex['system'][:80]}...")
        print(f"User   : {ex['user'][:100]}")
        print(f"Asst   : {ex['assistant'][:200]}...")
        print(f"Formatted length: {len(format_for_training(ex))} chars\n")

    # Save
    builder.save("/home/claude/mindbridge/sft/sft_train.jsonl", split="train")
