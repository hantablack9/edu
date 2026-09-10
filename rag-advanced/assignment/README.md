# RAG++ course assignment

This Codespaces-ready project implements the required Weave evaluation comparison for the RAG++ course.

It evaluates:

- retrieval with IR metrics and an LLM retrieval judge;
- response quality with traditional NLP metrics;
- correctness, relevance, and faithfulness with LLM judges;
- a single end-to-end RAG model whose traces and evaluation results are logged to Weave.

## Run in GitHub Codespaces

1. Fork wandb/edu to your GitHub account, then open the fork in Codespaces.
2. In the Codespace terminal, set secrets without committing them:

   ```bash
   export WANDB_API_KEY="..."
   export COHERE_API_KEY="..."
   export WANDB_ENTITY="your-wandb-entity"
   export WANDB_PROJECT="rag-course-assignment"
   ```

3. Run the evaluator:

   ```bash
   cd rag-advanced
   python assignment/run_evaluation.py
   ```

4. Open the printed Weave project URL and compare the logged evaluations. Copy that URL into assignment_submission.txt, then upload that one file in the course page.

The optional 70 percent Wandbot challenge is intentionally separate from the required assignment.
