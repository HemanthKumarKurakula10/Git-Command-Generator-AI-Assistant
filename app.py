# 📦 Required libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import torch

# 🤖 Load Phi-2 model and tokenizer
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# 💬 Text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 🧠 Helper: Format command + explanation
def format_git_response(command, explanation):
    return f"🔧 Git Command:\n```bash\n{command}\n```\n\n💡 Explanation: {explanation}"

# 🔍 Intelligent intent detection for common Git tasks
def get_fixed_git_command(user_input):
    user_input_clean = user_input.lower().strip()

    if "stash" in user_input_clean or "temporarily save" in user_input_clean or "save without commit" in user_input_clean:
        return format_git_response(
            "git stash",
            "Temporarily saves (stashes) your uncommitted changes so you can work on something else and come back to them later."
        )

    if any(kw in user_input_clean for kw in ["undo last commit", "keep changes", "soft reset"]):
        return format_git_response(
            "git reset --soft HEAD~1",
            "This undoes the last commit but keeps your changes staged (ready to recommit)."
        )

    if any(kw in user_input_clean for kw in ["undo last commit", "unstaged", "mixed reset"]):
        return format_git_response(
            "git reset HEAD~1",
            "This undoes the last commit and keeps your changes in your working directory (not staged)."
        )

    if "delete last commit" in user_input_clean or "discard changes" in user_input_clean:
        return format_git_response(
            "git reset --hard HEAD~1",
            "This removes the last commit and deletes all related changes permanently."
        )

    if "new branch" in user_input_clean and "switch" in user_input_clean:
        return format_git_response(
            "git checkout -b <branch-name>",
            "Creates a new branch and switches to it. Replace `<branch-name>` with your desired name."
        )

    if "push" in user_input_clean and "remote" in user_input_clean:
        return format_git_response(
            "git push origin <branch-name>",
            "Pushes your local branch to the remote repo. Replace `<branch-name>` with your actual branch."
        )

    if "clone" in user_input_clean:
        return format_git_response(
            "git clone <repo-url>",
            "Clones a remote Git repository to your local machine. Replace `<repo-url>` with the actual GitHub URL."
        )

    if "view history" in user_input_clean or "commit history" in user_input_clean:
        return format_git_response(
            "git log --oneline",
            "Shows a simplified list of commits (one per line)."
        )

    if "commit recent changes" in user_input_clean or "save changes" in user_input_clean or "commit" in user_input_clean:
        return format_git_response(
            'git commit -m "your message here"',
            "Commits the staged changes with a custom message. Replace the quoted text with your commit message."
        )

    return None  # Fallback to Phi-2

# 🧠 Assistant logic
def generate_git_command(user_input):
    # First, try rule-based intent matching
    fixed_response = get_fixed_git_command(user_input)
    if fixed_response:
        return fixed_response

    # Otherwise, use the model to generate a response
    prompt = f"Instruct: Convert this request into a Git command and explain it briefly:\n\"{user_input}\"\nOutput:"
    response = generator(prompt, max_new_tokens=200, temperature=0.4, do_sample=False)
    return response[0]["generated_text"].split("Output:")[-1].strip()

# 🎨 Gradio UI
gr.Interface(
    fn=generate_git_command,
    inputs=gr.Textbox(
        lines=2,
        placeholder="e.g. I want to stash my changes temporarily",
        label="📝 What do you want to do with Git?"
    ),
    outputs=gr.Textbox(
        lines=8,
        label="💡 Git Command + Explanation"
    ),
    title="🧑‍💻 Git Command Generator AI Assistant",
    description="Type your Git-related need in natural language, and get the correct command with explanation. Powered by Phi-2.",
    examples=[
        ["Undo last commit but keep changes"],
        ["Undo last commit but keep changes unstaged"],
        ["Delete last commit and changes"],
        ["Create a new branch and switch to it"],
        ["Push changes to remote"],
        ["Clone a repo from GitHub"],
        ["View commit history"],
        ["Want to commit recent changes"],
        ["Temporarily save my changes without committing"]
    ],
    theme="default",
    allow_flagging="never"
).launch()