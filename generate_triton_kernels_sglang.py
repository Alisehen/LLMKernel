import os
import re
import openai
import concurrent.futures

# --- Configuration ---
KERNEL_BENCH_DIR = '/home/hyc/LLMKernel/KernelBench'
GENERATED_DIR = '/home/hyc/generated_kernels'
SGLANG_SERVER_URL = "http://localhost:8001/v1" 
CONCURRENT_REQUESTS = 8 # Set concurrency level to 8
# ASSUMPTION: An sglang server is running locally and is compatible with the OpenAI API.
# You can start one with a command like: 
# python -m sglang.launch_server --model-path <your_model_path> --port 30000

# --- Prompt Templates (as provided by the user) ---
PROBLEM_STATEMENT = """You are given a pytorch function, and your task is to write the same triton implementation for it.
The triton implementation should change the name from Model to ModelNew, and have same input and output as the pytorch function."""
PROBLEM_INSTRUCTION = """Optimize the architecture with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no input and init function, no other text, and NO testing code! **Remember to Name your optimized output architecture ModelNew, do not use Model again!**"""

def generate_for_file(client, file_path, output_path):
    """
    Reads a file, generates the Triton kernel using the model, and saves it.
    """
    print(f"--- Processing: {file_path} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            arc_src = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Construct the prompt
    prompt = f"""{PROBLEM_STATEMENT} {PROBLEM_INSTRUCTION}
Now, you need to write the triton implementation for the following pytorch code:
```
{arc_src}
``` """

    messages = [{"role": "user", "content": prompt}]

    try:
        # Call the SGLang server
        response = client.chat.completions.create(
            model="default",  # Model name is often 'default' or ignored by local servers
            messages=messages,
            temperature=0.0,
            max_tokens=10240, # Increased max_tokens for potentially large kernels
        )
        
        generated_code = response.choices[0].message.content

        # Clean up the output - handle <think>...</think><answer>...</answer> format
        # Remove <think>...</think> block
        generated_code = re.sub(r'<think>.*?</think>', '', generated_code, flags=re.DOTALL).strip()

        # Extract content from <answer>...</answer> block if present
        answer_match = re.search(r'<answer>(.*?)</answer>', generated_code, flags=re.DOTALL)
        if answer_match:
            generated_code = answer_match.group(1).strip()

        # Remove markdown code blocks
        code_block_match = re.search(r'```(?:python)?\s*(.*?)```', generated_code, flags=re.DOTALL)
        if code_block_match:
            generated_code = code_block_match.group(1).strip()
        else:
            # Fallback: old cleanup method
            if generated_code.startswith("```python"):
                generated_code = generated_code[len("```python"):].strip()
            if generated_code.startswith("```"):
                generated_code = generated_code[3:].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3].strip()

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the generated kernel
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)
            
        print(f"Successfully generated and saved kernel to: {output_path}")

    except Exception as e:
        print(f"Error during API call or file writing for {file_path}: {e}")


def main():
    """
    Main function to iterate through KernelBench and generate Triton kernels concurrently.
    """
    print("Starting Triton kernel generation script with concurrency...")
    
    # Initialize OpenAI client to connect to the SGLang server
    client = openai.OpenAI(
        base_url=SGLANG_SERVER_URL,
        api_key="EMPTY",  # SGLang doesn't require a real key
    )

    # First, collect all the file paths to process
    tasks = []
    for root, _, files in os.walk(KERNEL_BENCH_DIR):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, KERNEL_BENCH_DIR)
                output_path = os.path.join(GENERATED_DIR, relative_path)
                tasks.append((full_path, output_path))

    # Use a ThreadPoolExecutor to process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # Create a lambda to pass the static client object to the worker threads
        future_to_task = {executor.submit(lambda p: generate_for_file(client, p[0], p[1]), task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()  # We can check for exceptions here if needed
            except Exception as exc:
                print(f'Task for {task[0]} generated an exception: {exc}')

    print("\nScript finished.")

if __name__ == "__main__":
    main()
