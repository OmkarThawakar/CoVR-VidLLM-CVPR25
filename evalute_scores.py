import numpy as np
import argparse


def eval_recall(scores_q2t):
    # Query->Target
    ranks = np.zeros(scores_q2t.shape[0])

    for index, score in enumerate(scores_q2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)  # type: ignore
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean3 = (tr1 + tr5 + tr10) / 3
    tr_mean4 = (tr1 + tr5 + tr10 + tr50) / 4

    eval_result = {
        "R1": round(tr1, 2),
        "R5": round(tr5, 2),
        "R10": round(tr10, 2),
        "R50": round(tr50, 2),
        "meanR3": round(tr_mean3, 2),
        "meanR4": round(tr_mean4, 2),
    }
    return eval_result

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval scores from similarity matrix file.")
    parser.add_argument(
        "--scores_file",
        type=str,
        default="sim_q2t.npy",
        help="Path to the file containing the similarity scores (NumPy .npy format)."
    )
    args = parser.parse_args()

    # Load the similarity scores from the file.
    # If your scores are saved as a CSV, you can use np.loadtxt instead.
    try:
        scores = np.load(args.scores_file)
    except Exception as e:
        print(f"Error loading scores file {args.scores_file}: {e}")
        return

    # Compute recall metrics.
    recalls = eval_recall(scores)

    # Print out the evaluation results.
    print("Evaluation Results:")
    for key, value in recalls.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
