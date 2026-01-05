
import argparse
from tqdm import tqdm
from dataset import FinVQAChartDataset
from models import (
    GPT4oModel, GPT4VModel, 
    ClaudeOpusModel, ClaudeSonnetModel,
    GeminiProModel, GeminiFlashModel,
    Qwen2VLModel, InternVL2Model, LLaVANext34BModel,
    CogVLM2Model, DeepSeekVLModel, InternLMXComposer2Model,
    GLM4VModel, MiniCPMVModel, Phi3VisionModel,
    LLaVAModel, QwenVLModel, InternVLModel, DePlotModel
)

def evaluate_model(model, dataset):
    """
    Evaluate VLM on FinVQA-Chart.
    Follows the protocol: Overall Accuracy, By Type, By Difficulty.
    """
    correct = 0
    results_by_type = {}
    results_by_difficulty = {}

    print(f"Starting evaluation for {model.model_name}...")
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        try:
            prediction = model.predict(
                image=sample['image'],
                question=sample['question'],
                options=sample.get('options')
            )
            
            # Ground truth handling
            gt = str(sample['correct_answer']).strip().lower()
            pred = str(prediction).strip().lower()
            
            # Simple containment check for fuzzy matching
            is_correct = (gt in pred) or (pred in gt)
            if sample.get('options') and len(pred) < 5: # Likely a letter answer A, B, C
                 is_correct = gt == pred
            
            if is_correct:
                correct += 1

            # Track by Type
            q_type = sample.get('question_type', 'unknown')
            if q_type not in results_by_type:
                results_by_type[q_type] = {'correct': 0, 'total': 0}
            results_by_type[q_type]['total'] += 1
            if is_correct:
                results_by_type[q_type]['correct'] += 1

            # Track by Difficulty
            diff = sample.get('difficulty', 'unknown')
            if diff not in results_by_difficulty:
                results_by_difficulty[diff] = {'correct': 0, 'total': 0}
            results_by_difficulty[diff]['total'] += 1
            if is_correct:
                results_by_difficulty[diff]['correct'] += 1

        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    overall_acc = correct / len(dataset) if len(dataset) > 0 else 0
    
    results = {
        'model': model.model_name,
        'overall_accuracy': overall_acc,
        'by_type': results_by_type,
        'by_difficulty': results_by_difficulty
    }
    return results

def print_results(results):
    print(f"\nResults for {results['model']}:")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    
    print("\nAccuracy by Question Type:")
    for qt, scores in results['by_type'].items():
        acc = scores['correct'] / scores['total'] if scores['total'] > 0 else 0
        print(f"  {qt}: {acc:.2%} ({scores['correct']}/{scores['total']})")
        
    print("\nAccuracy by Difficulty:")
    for diff, scores in results['by_difficulty'].items():
        acc = scores['correct'] / scores['total'] if scores['total'] > 0 else 0
        print(f"  {diff}: {acc:.2%} ({scores['correct']}/{scores['total']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_choices = [
        "gpt4o", "gpt4v", 
        "claude_opus", "claude_sonnet", 
        "gemini_pro", "gemini_flash", 
        "qwen2_vl", "internvl2", "llava_next", 
        "cogvlm2", "deepseek_vl", "internlm_xcomposer2", 
        "glm4v", "minicpm_v", "phi3_vision"
    ]
    parser.add_argument("--model", type=str, default="gpt4o", choices=model_choices, help="Model to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    args = parser.parse_args()

    ds = FinVQAChartDataset(split=args.split)

    if args.model == "gpt4o":
        model = GPT4oModel()
    elif args.model == "gpt4v":
        model = GPT4VModel()
    elif args.model == "claude_opus":
        model = ClaudeOpusModel()
    elif args.model == "claude_sonnet":
        model = ClaudeSonnetModel()
    elif args.model == "gemini_pro":
        model = GeminiProModel()
    elif args.model == "gemini_flash":
        model = GeminiFlashModel()
    elif args.model == "qwen2_vl":
        model = Qwen2VLModel()
    elif args.model == "internvl2":
        model = InternVL2Model()
    elif args.model == "llava_next":
        model = LLaVANext34BModel()
    elif args.model == "cogvlm2":
        model = CogVLM2Model()
    elif args.model == "deepseek_vl":
        model = DeepSeekVLModel()
    elif args.model == "internlm_xcomposer2":
        model = InternLMXComposer2Model()
    elif args.model == "glm4v":
        model = GLM4VModel()
    elif args.model == "minicpm_v":
        model = MiniCPMVModel()
    elif args.model == "phi3_vision":
        model = Phi3VisionModel()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_model()
    results = evaluate_model(model, ds)
    print_results(results)
