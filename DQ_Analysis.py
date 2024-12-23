import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ollama
from sklearn.preprocessing import LabelEncoder
import io

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "active_section" not in st.session_state:
    st.session_state.active_section = "Upload File"

# Define functions for each section
def upload_file():
    """Upload and display the dataset."""
    st.header("Upload File")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("File loaded successfully!")
            st.dataframe(st.session_state.df)
        except Exception as e:
            st.error(f"Error occurred while loading the file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

def data_review():
    """Preview and review the dataset."""
    st.header("Data Info")
    if st.session_state.df is not None:
        df = st.session_state.df

        # Show Data Info
        # st.write("### Dataset Information")
        if st.button("Show Data"):
            st.write(df)
        if st.button("Show Columns"):
            st.write(df.columns.tolist())
        if st.button("Show Dimensions"):
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        if st.button("Show Describe"):
            st.write(df.describe())
    else:
        st.warning("Please upload a dataset first!")

def handle_missing_values():
    """Handle missing values in the dataset."""
    st.header("Handle Missing Values")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        # Checkbox to show missing values before cleaning
        if st.checkbox("Show Missing Values"):
            st.write("### Missing Values Before Cleaning")
            st.write(df.isnull().sum())

        # Handle missing values
        col_to_handle = st.selectbox("Select a column to handle missing values", df.columns)
        handle_option = st.radio(
            "Select a method to handle missing values",
            ("Mode", "Mean", "Median","Drop the Column")
        )
        if st.button("Apply Cleaning"):
            if handle_option == "Mode":
                mode_value = df[col_to_handle].mode()[0]
                st.success(f"Mode Value for '{col_to_handle}': {mode_value}")
                df[col_to_handle].fillna(mode_value, inplace=True)

            elif handle_option == "Mean":
                    mean_value = df[col_to_handle].mean()
                    st.success(f"Mean Value for '{col_to_handle}': {mean_value}")
                    df[col_to_handle].fillna(mean_value, inplace=True)

            elif handle_option == "Median":
                median_value = df[col_to_handle].median()
                st.success(f"Median Value for '{col_to_handle}': {median_value}")
                df[col_to_handle].fillna(median_value, inplace=True)

            elif handle_option == "Drop the Column":
                    df.drop(columns=[col_to_handle], inplace=True)
                    st.success(f"Column '{col_to_handle}' has been dropped.")

            # Update session state with cleaned data
            st.session_state.df = df

        # Checkbox to display cleaned data
        if st.checkbox("Show Data After Cleaning"):
            st.write("### Data After Cleaning")
            st.write(df.isnull().sum())
            st.dataframe(st.session_state.df)

    else:
        st.warning("Please upload a dataset first!")

def handle_duplicates():
    """Handle duplicate rows in the dataset."""
    st.header("Handle Duplicates")
    if st.session_state.df is not None:
        df = st.session_state.df

        # Display the number of duplicates before cleaning
        duplicate_count_before = df.duplicated().sum()
        st.write(f"### Number of Duplicate Rows Before Cleaning: {duplicate_count_before}")

        # Remove duplicates if the user clicks the button
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state.df = df
            duplicate_count_after = df.duplicated().sum()
            st.success(f"Duplicates removed. Current shape: {df.shape}")
            st.write(f"### Number of Duplicate Rows After Cleaning: {duplicate_count_after}")
            st.write("### Data After Removing Duplicates")
            st.dataframe(df)
    else:
        st.warning("Please upload a dataset first!")


def handle_correlation():
    """Handle correlation calculation for numeric columns and plot a heatmap."""
    st.header("Correlation Matrix")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Filter only numeric columns
        numeric_df = df.select_dtypes(include=np.number)

        if not numeric_df.empty:
            # Calculate correlation matrix
            correlation_matrix = numeric_df.corr()

            # Create the figure and axis explicitly
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot correlation matrix using seaborn heatmap
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)

            # Display the plot
            st.pyplot(fig)  # Now passing the figure object to st.pyplot()
        else:
            st.warning("No numeric columns found to compute correlation.")
    else:
        st.warning("Please upload a dataset first!")
        
def handle_outliers():
    """Handle outliers by either dropping the row or clipping values."""
    st.header("Handle Outliers")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            st.write("### Numeric Columns:")
            st.write(numeric_cols)

            selected_col = st.selectbox("Select a column to handle outliers", numeric_cols)

            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_befor = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)].shape[0]
            st.write(f"Number of outliers in  {selected_col} Before Handling  : {outliers_befor}")
            st.write(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

            method = st.radio(
                "Select a method to handle outliers:",
                ("Drop the Row", "Clip Outliers")
            )

            if st.button("Apply Outlier Handling"):
                if method == "Drop the Row":
                    df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                    st.success(f"Rows with outliers in column '{selected_col}' have been dropped.")
                elif method == "Clip Outliers":
                    df[selected_col] = df[selected_col].clip(lower=lower_bound, upper=upper_bound)
                    st.success(f"Outliers in column '{selected_col}' have been clipped.")

                outliers_after = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)].shape[0]
                st.write(f"Number of outliers in  {selected_col} After Handling  : {outliers_after}")

                st.session_state.df = df
                st.write("### Updated Data:")
                st.dataframe(df)
        else:
            st.warning("No numeric columns available to process.")
    else:
        st.warning("Please upload a dataset first!")



def data_visualization():
    """Perform general analysis on the dataset."""
    st.header("Dataset Visualization")
    if st.session_state.df is not None:
        df = st.session_state.df

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols.insert(0, "Select")

        col_to_analyze = st.selectbox("Select a column for Visualization", numeric_cols)

        if col_to_analyze != "Select":  
            # Box Plot for the selected column
            st.write(f"### Box Plot for {col_to_analyze}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col_to_analyze], color="skyblue", ax=ax)
            st.pyplot(fig)


            # Histogram for the selected column
            st.write(f"### Histogram for {col_to_analyze}")
            fig, ax = plt.subplots()
            ax.hist(df[col_to_analyze].dropna(), bins=30, color="skyblue", edgecolor="black")
            st.pyplot(fig)
        else:
            st.info("Please select a column to perform virsualization.")  

    else:
        st.warning("Please upload a dataset first!")



def data_type_conversion():
    """Convert data types of columns."""
    st.header("Data Type Conversion")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Show the current data types of the columns
        st.write("### Current Data Types")
        st.write(df.dtypes)

        # Select a column to convert its data type
        col_to_convert = st.selectbox("Select a column to convert", df.columns)

        # Select the target data type
        data_type = st.selectbox(
            "Select the target data type",
            ["int", "float", "str", "bool", "datetime"]
        )

        if st.button("Apply Conversion"):
            try:
                if data_type == "int":
                    df[col_to_convert] = df[col_to_convert].astype(int, errors="raise")
                elif data_type == "float":
                    df[col_to_convert] = df[col_to_convert].astype(float, errors="raise")
                elif data_type == "str":
                    df[col_to_convert] = df[col_to_convert].astype(str, errors="raise")
                elif data_type == "bool":
                    df[col_to_convert] = df[col_to_convert].astype(bool, errors="raise")
                elif data_type == "datetime":
                    df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors="raise")

                st.session_state.df = df
                st.success(f"Column '{col_to_convert}' converted to {data_type} successfully!")

            except Exception as e:
                st.error(f"Error occurred: {e}")

        # Option to display the updated dataframe and data types
        if st.checkbox("Show Updated DataFrame"):
            st.write("### Data After Conversion")
            st.write(df.dtypes)
            st.dataframe(df)
    else:
        st.warning("Please upload a dataset first!")

def chat_Using_rag():
        st.subheader("Chat with Dataset using RAG")

        def ollama_generate(query: str, model: str = "llama3.2:1b") -> str:
            """Generate a response using Ollama."""
            try:
                result = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
                return result.get("message", {}).get("content", "No response content.")
            except Exception as e:
                return f"Error: {e}"

        # Function to chat with CSV using Ollama
        def chat_with_csv_ollama(df, prompt, model="llama3.2", max_rows=10):
            """Chat with a CSV using Ollama."""
            # Summarize dataset: Include column names, row count, and sample rows
            summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
            column_info = "Columns:\n" + "\n".join([f"- {col} (type: {str(df[col].dtype)})" for col in df.columns])
            sample_data = f"Sample rows:\n{df.head(5).to_string(index=False)}"

            # Include data content (limit rows if necessary)
            data_content = f"The dataset:\n{df.head(max_rows).to_string(index=False)}"

            # Create the query
            query = f"""
            You are a data assistant. Here is the summary of the dataset:
            {summary}
            {column_info}
            {sample_data}

            {data_content}

            Based on this dataset, answer the following question:
            {prompt}
            """
            
            # Use the `ollama_generate` function to get the response
            return ollama_generate(query, model=model)

        # Initialize session state for query and response history
        if "conversation" not in st.session_state:
            st.session_state.conversation = []  # Stores the history as a list of dictionaries with roles and messages

        # App title
        st.title("ChatCSV powered by Ollama")

        # Upload CSV section
            

        if st.session_state.df is not None:
            # Read and display the CSV
            st.info("CSV Uploaded Successfully")
            
            st.dataframe(st.session_state.df, use_container_width=True)

            # Chat interface
            st.info("Chat Below")
            user_input = st.chat_input("Ask a question:")

            if user_input:
                # Add user query to the conversation
                st.session_state.conversation.append({"role": "user", "content": user_input})

                # Generate response from Ollama
                with st.spinner("Generating response..."):
                    assistant_response = chat_with_csv_ollama(st.session_state.df, user_input)

                # Add assistant response to the conversation
                st.session_state.conversation.append({"role": "assistant", "content": assistant_response})

            # Display the conversation
            for message in st.session_state.conversation:
                if message["role"] == "user":
                    st.chat_message("user").markdown(message["content"])
                elif message["role"] == "assistant":
                    # Check if the message contains code blocks
                    if "```" in message["content"]:
                        # Split by code blocks
                        code_blocks = message["content"].split("```")
                        for i, block in enumerate(code_blocks):
                            if i % 2 == 1:  # Odd indices are code blocks
                                st.code(block.strip(), language="python")  # Render as code
                            else:
                                if block.strip():  # Avoid rendering empty text
                                    st.chat_message("assistant").markdown(block.strip())
                    else:
                        st.chat_message("assistant").markdown(message["content"])




def download_csv():
    """Allow users to download the dataset as a CSV file."""
    if st.session_state.df is not None:
        # Convert dataframe to CSV
        csv = st.session_state.df.to_csv(index=False)
        # Convert to bytes so it can be downloaded
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data available to download.")

# Sidebar navigation buttons
st.sidebar.title("Data Quality Analysis (Project Team 18)")
if st.sidebar.button("Upload File"):
    st.session_state.active_section = "Upload File"
if st.sidebar.button("Data Info"):
    st.session_state.active_section = "Data Review"
if st.sidebar.button("Handle Missing Values"):
    st.session_state.active_section = "Handle Missing Values"
if st.sidebar.button("Handle Duplicates"):
    st.session_state.active_section = "Handle Duplicates"
if st.sidebar.button("Handle Outliers"):
    st.session_state.active_section = "Handle Outliers"
if st.sidebar.button("Correlation Matrix"):
    st.session_state.active_section = "Correlation Matrix"
if st.sidebar.button("data_visualization"):
    st.session_state.active_section = "data_visualization"
if st.sidebar.button("Data Type Conversion"):
    st.session_state.active_section = "Data Type Conversion"
if st.sidebar.button("Chat Using RAG"):
    st.session_state.active_section = "Chat Using RAG"
if st.sidebar.button("Download CSV"):
    st.session_state.active_section = "Download CSV"

# Navigation logic
if st.session_state.active_section == "Upload File":
    upload_file()
elif st.session_state.active_section == "Data Review":
    data_review()
elif st.session_state.active_section == "Handle Missing Values":
    handle_missing_values()
elif st.session_state.active_section == "Handle Duplicates":
    handle_duplicates()
elif st.session_state.active_section == "Handle Outliers":
    handle_outliers()
elif st.session_state.active_section == "Correlation Matrix":
    handle_correlation()
elif st.session_state.active_section == "data_visualization":
    data_visualization()
elif st.session_state.active_section == "Chat Using RAG":
    chat_Using_rag()
elif st.session_state.active_section == "Data Type Conversion":
    data_type_conversion()
elif st.session_state.active_section == "Download CSV":
    download_csv()
