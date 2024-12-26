from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.meteor_score import meteor_score
from Levenshtein import distance as levenshtein_distance
from bert_score import score
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="LLM Evaluation API",
    description="Evaluate LLM outputs using METEOR, Levenshtein Distance, and BERTScore.",
    version="1.0.0"
)

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Pydantic Models for request and response
class EvaluationRequest(BaseModel):
    references: List[str]
    inputs: List[str]


class EvaluationResponse(BaseModel):
    meteor: float
    levenshtein_similarity: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float


# Helper function to compute metrics
def compute_metrics(references: List[str], predictions: List[str]) -> dict:
    # METEOR Scores
    meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(references, predictions)]
    print(f"METEOR Scores: {meteor_scores}")  # Print METEOR scores
    
    # Levenshtein Distance (normalized similarity)
    levenshtein_scores = [
        1 - levenshtein_distance(ref, pred) / max(len(ref), len(pred))
        if len(ref) > 0 and len(pred) > 0 else 0
        for ref, pred in zip(references, predictions)
    ]
    print(f"Levenshtein Scores: {levenshtein_scores}")  # Print Levenshtein scores

    # BERTScore
    P, R, F1 = score(predictions, references, lang="en", verbose=False)
    print(f"BERTScore Precision: {P}")
    print(f"BERTScore Recall: {R}")
    print(f"BERTScore F1: {F1}")

    # Aggregate Metrics
    return {
        "meteor": np.mean(meteor_scores),
        "levenshtein_similarity": np.mean(levenshtein_scores),
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


# Endpoint to evaluate model outputs
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest):
    if len(request.references) != len(request.inputs):
        raise HTTPException(
            status_code=400, detail="The number of references must match the number of inputs."
        )

    references = request.references
    inputs = request.inputs
    predictions = []

    # Generate predictions
    for input_text in inputs:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=128, num_return_sequences=1)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        print(f"Input: {input_text}")
        print(f"Prediction: {prediction}")

    # Compute metrics
    metrics = compute_metrics(references, predictions)

    return EvaluationResponse(**metrics)


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Basic eval model."}

