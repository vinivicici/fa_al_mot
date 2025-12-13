import pandas as pd
import torch
from transformers import pipeline
import warnings
import json

# Try to import tqdm for a progress bar, but don't fail if it's not installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm library not found. Progress bar will not be shown.")
    print("Install with: pip install tqdm")
    # Define a dummy tqdm function if it's not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

# --- Configuration ---
# Use the "Instruct" model for following instructions
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

# The name of your input CSV file (Updated to match your upload)
CSV_FILE = "merged_dataset.csv" 

# The name of the NEW, FINAL CSV file where results will be saved
OUTPUT_CSV_FILE = "dataset_with_llm.csv"

# --- REVISED INSTRUCTION ---
# This instruction is now extremely specific about how to find the brand.
TASK_INSTRUCTION = (
    "You are a meticulous data extraction engine. Your task is to analyze the product description "
    "and return ONLY a single, structured text block. You must extract *all* relevant details and omit nothing. "
    "The output must follow this exact format, with 'N/A' for any missing information:\n\n"
    "Brand: [Find the brand. Look for text like 'brand: HNM' or 'brandUserProfile: ARROW']\n"
    "Main Item: [The primary product described]\n"
    "Materials: [A list of all mentioned materials (e.g., 100% cotton, 64% polyester)]\n"
    "Care: [A list of all care instructions (e.g., Machine wash cold, Do not bleach)]\n"
    "Style/Fit: [Any style, fit, or usage recommendations (e.g., Straight leg, Comfort Fit)]\n"
    "Other Details: [A list of any other important details (e.g., Concealed zipper, Model's Statistics)]\n\n"
    "Do not include any text, apologies, or explanations before or after this structured block."
)
# --- End Configuration ---

def load_model_pipeline(model_id):
    """
    Loads the Hugging Face model into a text-generation pipeline.
    
    This requires you to be logged into Hugging Face locally.
    Run `huggingface-cli login` in your VSCode terminal first.
    """
    print(f"Loading model: {model_id}...")
    print("This may take several minutes and download many gigabytes.")
    
    try:
        # device_map="auto" will use GPU (CUDA) if available
        # torch_dtype="auto" or torch.bfloat16 speeds up inference
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device_map="auto"
        )
        print("Model loaded successfully.")
        return pipe
    except ImportError:
        print("\n--- ERROR ---")
        print("Could not load model. Did you install all requirements?")
        print("Run: pip install transformers torch accelerate pandas tqdm")
        print("-------------")
        return None
    except Exception as e:
        print(f"\n--- ERROR ---  \n{e}")
        print("\nCould not load model. Did you log in?")
        print("Run `huggingface-cli login` in your terminal.")
        print("-------------")
        return None

def run_inference(pipe, description, instruction):
    """
    Formats the prompt using the Llama 3 Instruct template and runs inference.
    """
    
    # Create the message list for the Llama 3 Instruct template
    messages = [
        {
            "role": "user",
            "content": f"{instruction}\n\n### Product Description:\n{description}"
        }
    ]

    # The pipeline handles applying the chat template automatically
    
    # Set terminators to stop the model cleanly
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Run the model
    outputs = pipe(
        messages,
        max_new_tokens=512,  # Max length of the model's answer
        eos_token_id=terminators,
        do_sample=False,     # Set to False for more deterministic, faster output
        temperature=0.1,   # Low temperature for extraction
        top_p=0.9,
    )
    
    # Get the assistant's response content
    response = outputs[0]["generated_text"][-1] 
    return response['content']

def main():
    # 1. Load the model
    pipe = load_model_pipeline(MODEL_ID)
    if pipe is None:
        print("Exiting due to model loading failure.")
        return

    # 2. Load the data
    print(f"Loading data from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        if 'description' not in df.columns:
            print("Error: 'description' column not found in the CSV.")
            return
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_FILE}")
        print("Make sure it's in the same folder as this script.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # --- RUNNING ON FULL FILE ---
    # The df.head(5) line has been removed to process the entire DataFrame.
    print(f"--- RUNNING IN FULL MODE ---")
    
    print(f"Found {len(df)} rows to analyze.")

    # 3. Format and run inference on all rows
    results_list = []
    
    # Use tqdm for a progress bar
    print("Analyzing descriptions...")
    for row in tqdm(df.itertuples(), total=len(df), desc="Analyzing"):
        description = row.description
        
        # Define the structure for missing data (one column)
        empty_data = {
            "llama_organized_description": None
        }

        if pd.isna(description):
            # Append empty data if description is missing
            results_list.append(empty_data)
            continue

        model_response = run_inference(pipe, str(description), TASK_INSTRUCTION)
        
        # No more JSON parsing. Just save the raw (but structured) text response.
        try:
            results_list.append({"llama_organized_description": model_response})
        
        except Exception as e:
            # Handle other potential errors
            print(f"\nAn unexpected error occurred for index {row.Index}: {e}")
            error_data = empty_data.copy()
            error_data["llama_organized_description"] = "GENERAL_ERROR"
            results_list.append(error_data)

    # 4. Save results to a new CSV
    print("\nAnalysis complete. Processing results...")
    
    # Convert the list of dictionaries into a DataFrame
    # This DataFrame will now have ONE column: "llama_organized_description"
    results_df = pd.DataFrame(results_list)
    
    # Reset index of original df to ensure a clean join
    df.reset_index(drop=True, inplace=True)
    
    # Join the new results DataFrame with the original DataFrame
    # This combines the original columns and the new single, structured column
    final_df = df.join(results_df)
    
    # Save the combined data to a new CSV file
    try:
        final_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Successfully saved final results to {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    # Suppress a common, benign warning from the transformers library
    warnings.filterwarnings(
        "ignore", 
        message=".*where max_length*.", 
        category=UserWarning
    )
    main()

