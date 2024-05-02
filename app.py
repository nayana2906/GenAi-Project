import os
import streamlit as st
import pandas as pd    
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv , find_dotenv
from apikey import apikey

os.environ['GOOGLE_API_KEY'] = apikey
load_dotenv(find_dotenv())

# Define the model ID for the Davinci model
model_id = "gpt-3.5-turbo-instruct"  # or any other supported model ID

# Initialize OpenAI with the new model ID

st.title('AI Assistant for Data Science')
st.write("Hello, I am your AI Assistant and I am here to help you with the data analysis")

with st.sidebar:
    st.write('''Your Data Science Adventure Begins with an CSV File.''')
    st.caption("That's why I'd love for you to upload a CSV file. Once we have your data in hand, we'll dive into understanding it ar Then, we'll work together to shape your business challenge into a c. I'll introduce you to the coolest machine learning models, and we.")

    st.divider()
    st.caption("<p style='text-align:center'> made by R V students</p>", unsafe_allow_html=True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    st.header('Exploratory Data Analysis Part')
    st.subheader('Solution')
    file_csv = st.file_uploader("upload your file here!!!", type="csv")
    
    if file_csv is not None:
        file_csv.seek(0)
        df = pd.read_csv(file_csv, low_memory=False)
        #llm model
        llm = OpenAI(model=model_id, temperature=0)
       
        @st.cache_data
        def steps_eda():
            steps_eda = llm('what are the steps of EDA')
            return steps_eda
        
        #pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("*Data Cleaning*")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does tha")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and")
            st.write(duplicates)

            st.write("Data Summarisation*")

            st.write(df.describe())

            correlation_analysis = pandas_agent.run("Calculate correlations bet")

            st.write(correlation_analysis)

            outliers = pandas_agent.run("Identify outliers in the data that may")

            st.write(outliers)

            new_features = pandas_agent.run("What new features would be interested in?")

            st.write(new_features)

            return
        
        @st.cache_data
        def function_question_variable():
            st.line_chart(df,y = [user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of{ user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"check for normality")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of the outliers")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends , seasonality")
            st.write(trends)
            
            return

        @st.cache_data
        def function_question_dataframe(user_question_dataframe):
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return
 
        #main 
        st.header('Exploratory data analysis')
        st.subheader('general information about the dataset')
        with st.sidebar:
            with st.expander('what are the steps of EDA'):
                st.write(steps_eda())  

        function_agent()

        st.subheader("variable of study")
        user_question_variable = st.text_input("what variable are you interested in ?")
        if user_question_variable is not None and user_question_variable != '' :
            function_question_variable()
            st.subheader ('Further Study')
        
        if user_question_variable : 
            user_question_dataframe = st.text_input("Is there anthing else you would like to know about the dataframe")
            if user_question_dataframe is not None and user_question_variable not in ('', 'no' ,"No"):
               function_question_dataframe(user_question_dataframe)
            if user_question_dataframe in ('no' , 'No'):
                st.write('')

