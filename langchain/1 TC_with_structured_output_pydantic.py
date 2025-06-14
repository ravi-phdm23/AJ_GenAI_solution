from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
import pandas as pd

# Load environment variables (e.g., for OpenAI API key)
load_dotenv()

# Initialize the LangChain model
model = ChatOpenAI()

# Define structured schema for a Basel 3.1 test case
class BaselTestCase(BaseModel):
    test_title: str = Field(description="Overall title describing the Basel 3.1 RWA requirement being tested")
    test_case_title: str = Field(description="Concise title of the specific test case")
    test_case_description: str = Field(description="Detailed description of what this test case is validating")
    test_steps: List[str] = Field(description="Step-by-step instructions to execute the test case")
    test_data: List[str] = Field(description="Test data inputs required for the test")
    expected_results: List[str] = Field(description="Expected outcome or result after performing the test steps")

# Wrap model with structured output
structured_model = model.with_structured_output(BaselTestCase)

# Example requirement prompt
prompt = """
Create a test case for the following Basel 3.1 requirement: 
"For real estate exposures, Loan-to-Value (LTV) ratio must be accurately computed based on the current market value and the outstanding loan balance. 
If LTV > 80%, apply a risk weight of 75%. If LTV <= 80%, apply a risk weight of 35%."

The test case should contain:
- A test title
- A test case title and description
- At least 3 test steps
- The necessary test data (e.g., market value, loan balance)
- Expected results for each case
"""

# Generate structured result
result = structured_model.invoke(prompt)
result_dict = result.dict()

# Normalize to row-wise DataFrame
max_len = max(len(result_dict['test_steps']), len(result_dict['test_data']), len(result_dict['expected_results']))
normalized_rows = []

for i in range(max_len):
    row = {
        'test_title': result_dict['test_title'],
        'test_case_title': result_dict['test_case_title'],
        'test_case_description': result_dict['test_case_description'],
        'step_number': f"Step {i+1}",
        'test_step': result_dict['test_steps'][i] if i < len(result_dict['test_steps']) else "",
        'test_data': result_dict['test_data'][i] if i < len(result_dict['test_data']) else "",
        'expected_result': result_dict['expected_results'][i] if i < len(result_dict['expected_results']) else "",
    }
    normalized_rows.append(row)

df_normalized = pd.DataFrame(normalized_rows)

# Save to CSV (optional)
df_normalized.to_csv("Output/normalized_basel_3_1_test_case.csv", index=False)

# Print DataFrame for inspection
print("\nNormalized Test Case Output:\n")
print(df_normalized)
