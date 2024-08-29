# -- IMPORTS --
# -- LANGCHAIN IMPORTS --
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import S3DirectoryLoader
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -- DOC READER - S3 IMPORTS --
import boto3

# -- DATA CLEANING & OS OPERATIONS --
import re
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# -- FUNCTIONS --

# -- NEWER CREATE RETRIEVAL CHAIN SETUP --

def new_retrieval_chain(vector_database, input_query: str):

    # repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    large_language_model = HuggingFaceEndpoint(
        repo_id=repo_id, temperature=0.3, token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    )

    system_prompt = (
        "You are an assistant that does not deviate from the instructions for answering a question. "
        "Use the following pieces of retrieved context to answer "
        "the question mentioned in the human prompt. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "It is very important that you only answer to the question given by the human and you should never make up your own question"
        "The structure of your answer should always be:"
        "Answer: answer to the question"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever = vector_database.as_retriever()

    question_answer_chain = create_stuff_documents_chain(large_language_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": input_query})
    return response["answer"]
    # print(response["context"])

# # -- RETRIEVAL QA CHAIN SETUP --
# def retrieval_qa_chain(vector_database, input_query: str):

#     repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#     # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     large_language_model = HuggingFaceEndpoint(
#         repo_id=repo_id, max_new_tokens=128, temperature=0.3, token=os.environ['HUGGINGFACEHUB_API_TOKEN']
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         large_language_model,
#         retriever=vector_database.as_retriever(
#             k = 1
#         ),  # K - Number of documents to fetch from the vector database
#         chain_type_kwargs={
#             "verbose": True  # uncomment for debugging - to see the input prompt along with the system message for the llm
#         }
#     )

#     llm_output = qa_chain.invoke(
#         input=input_query
#     )  # invoke / call the qa chain with the input query

#     return llm_output['result'] # return only the result field of the llm-output

# -- SEMANTIC DUPLICATE REMOVAL --
def remove_semantic_duplicates(sentences: list):
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    paraphrases = util.paraphrase_mining(model, sentences)

    uniques = []
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        #print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))
        if(score > 0.8):
            uniques.append(sentences[i])
        
    for i in range(len(sentences) - 1, -1, -1):
        if sentences[i] in uniques:
            sentences.pop(i)

    return sentences
 
# -- LOAD DOCUMENTS FROM S3 BUCKET --
def load_documents_from_s3_directory(directory_name, prefix):

    """Loads files (all types) from the specified S3 directory.

    Args:
        directory_name: The name of the S3 directory.
        prefix: Prefix of the specified S3 directory.

    Response:
        List of Loaded documents from the S3 directory.
    """


    document_loader = S3DirectoryLoader(
        directory_name, prefix = prefix)
    
    response = document_loader.load()

    return response

# -- TEXT SPLITTER HELPER FUNCTION --
def chunk_data(documents, chunk_size, chunk_overlap):

    """ Helper function that splits the incoming documents into chunks for further processing.
    
    Args: 
        documents (List): Raw documents loaded from the document loader.
        chunk_size (int): Size of a particular chunk for splitting 
        chunk_overlap (int): Overlap to maintain context between each chunk.
    """

    documents[0].page_content = re.sub(r'\s{3,}', ' ', documents[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    return documents

# -- CREATING VECTOR STORE FROM DOCUMENTS AND INDEX NAME --
def create_vector_store_from_docs(documents, index_name):
 
    """For the documents from the parameter, create a vector store for context retrieval
   
    Args:
        documents (list): A list of strings which contain the document contents
 
    Returns:
        vector store: A vector store which has the context - further used in the retrieval QA chain
    """
 
    Pinecone(api_key=os.environ['PINECONE_API_KEY'])
 
    # index_name = "docs-rag-chatbot"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        documents,
        index_name=index_name,
        embedding=embeddings
    )
 
    return vectorstore_from_docs

# -- WRITING RESPONSE OF QUESTIONNAIRE BACK TO S3 BUCKET --
def write_to_s3_folder(bucket_name, folder_name, file_name, content):
  """Writes content to a specific folder in an S3 bucket using put_object.

  Args:
    bucket_name: The name of the S3 bucket.
    folder_name: The name of the folder in the bucket.
    file_name: The name of the file.
    content: The content to be written to the file.
  """

  s3 = boto3.client('s3')
  object_key = f"{folder_name}/{file_name}"

  try:
    s3.put_object(Body=content, Bucket=bucket_name, Key=object_key)
    print(f"File uploaded successfully to S3: s3://{bucket_name}/{object_key}")
  except Exception as e:
    print(f"Error uploading file to S3: {e}")

# -- FETCH ALL THE QUESTIONS FROM THE RAW TEXT USING LLM CALL --
def fetch_questions_from_document(document):
 
    """For the given document in the argument, retrieve the questions and append to a string / list
 
    Args:
        document (list) : The document from which the questions needs to be extracted
 
    Return:
        list: A list of questions
 
    """
    file_data_string = document
 
    repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
 
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        # max_new_tokens=4096,
        temperature=0.2,
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
    )
 
    instruction = "Retrieve all the questions from the text given in the Document while making sure not to generate any duplicates. If there is follow-up questions to a given question, make sure it is included in the same line and not with line breaks. Make sure while returning the question, each question has two linebreaks between."
    context = file_data_string
 
    template = """Instruction: {instruction}
 
    Document: {context}
 
    Answer: Give out only the list of questions and nothing else."""
 
    prompt = template.format(instruction=instruction, context=context)
    file_question_content = llm(prompt)

    return file_question_content

# -- FETCH ANSWER FOR QUESTIONS USING LLM RAG - RETRIEVAL QA CHAIN
def fetch_result_from_context(question_list, vector_store):
 
    """For all the questions present in the document, fetch answer from the context and append to a list
   
    Args:
        document (string): A big string containing all the questions
 
    Returns:
        string: A big string which contains questions followed by answer
    """
 
    # vector_store = []
    result = ""
 
    for question in question_list:
        query = f"""
        {question}
        """

        result += f"""Question: {question}"""
        result += '\n'
        result += new_retrieval_chain(vector_store, query)
        result += '\n\n'
       
    return result


# embedding_function = HuggingFaceEmbeddings()

# index_name = "docs-rag-chatbot"

# vector_database = PineconeVectorStore(  
#     index_name = index_name, embedding=embedding_function
# ) 

# questions = [
#     "1. Describe your application's architecture and different tiers.",
#     "2. Describe your coding practices.",
#     "3. How do you test your application?",
#     "4. Do you perform web application vulnerability testing? What is the frequency?"
# ]

# result = fetch_result_from_context(questions, vector_database)
# print(result)
# with open("test1.txt", 'a', encoding="utf-8") as f:
#     f.write(result)
#     f.write('\n')


def main():

    # -- CREATING VECTOR STORE --
    # print('call started')
    # bucket_name = 'questionaire-context-docs'
    # prefix = 'documents/'
    # context_file_data = load_documents_from_s3_directory(bucket_name, prefix)
    # print("reading context done..!")
    # # Split the documents into chunks and create a vector store from the chunks
    # documents = chunk_data(context_file_data, 3500, 100)
    # index_name = 'docs-rag-chatbot'
    # vector_store = create_vector_store_from_docs(documents, index_name)
    # print("vector db created..!")

    embedding_function = HuggingFaceEmbeddings()

    index_name = "docs-rag-chatbot"

    vector_store = PineconeVectorStore(  
        index_name = index_name, embedding=embedding_function
    ) 

    # -- FETCHING QUESTIONS FROM DOCS --
    bucket_name = 'questionaire-query-docs'
    prefix = 'questions/'
    questionnaires = load_documents_from_s3_directory(bucket_name, prefix)
    print("reading questionnaire done..!")
    print("LEN :::", len(questionnaires))
    # with open("a.txt", 'w') as f:
    #     f.write(str(questionnaires))
    print(questionnaires[0])

    # -- FETCH RESULTS -- 
    for questionnaire in questionnaires:

        chunks = chunk_data([questionnaire], 3500, 10)
        print('chunks created!')
        print("LEN CHUNKS :::", len(chunks))

        master_question_list = []

        for chunk in chunks:
            # print(chunks[0].page_content)
            # print(chunks[0])
            questions = ''
            questions = fetch_questions_from_document(chunk.page_content) 
            questions += "\n\n"
            questions_list = [question.strip() for question in questions.strip().split('\n\n')]
            questions_list = remove_semantic_duplicates(questions_list)

            for question in questions_list:
                master_question_list.append(question)

        for question in master_question_list:
            with open("ques.txt", 'a', encoding="utf-8") as f:
                f.write(question)
                f.write('\n\n')

        result_folder_name = "result"
        result_file_name = questionnaire.metadata["source"].split("/")[-1].split(".")[0]
 
        print(master_question_list)

    # # questions = [
    # #     "Describe your application's architecture and different tiers.",
    # #     "Describe your coding practices.",
    # #     "How do you test your application?",
    # #     "Do you perform web application vulnerability testing? What is the frequency?",
    # #     "If a vulnerability is identified in the software, what is the time required for fixing?",
    # #     "What is the uptime SLA for the software?",
    # #     "What is the DDOS handling capacity of the application",
    # #     "API keys should be managed and stored securely, public access to the API keys should be prohibited and Key rotation frequency needs to be defined.",
    # #     "Deploy use of Single-Sign-On (SSO) between on-premises systems and Cloud environment.",
    # #     "SaaS application should support the following- SAML, MFA, OTP(not needed if MFA being implemented) & Oauth 2.0",
    # #     "Do you use industry standards (Build Security in Maturity Model [BSIMM] benchmarks, Open Group ACS Trusted Technology Provider Framework, NIST, etc.) to build in security for your Systems/Software Development Lifecycle (SDLC)?", 

    # #     "Do you verify that all of your software/software components suppliers (if any) adhere to industry standards for Systems/Software Development Lifecycle (SDLC) security?",
    # #     "Do you review your applications for security vulnerabilities (E.g: OWASP top-10) and address any issues prior to deployment to production?",
    # #     "Do you use an automated source code analysis tool to detect security defects in code prior to production?",
    # #     "Do you use manual source-code analysis to detect security defects in code prior to production?", 

    # #     "All production, development, QA, testing environment should be segregated. Access to production systems should be allowed as per the business-requirment.",
    # #     "Are data input and output integrity routines (i.e., reconciliation and edit checks) implemented for application interfaces and databases to prevent manual or systematic processing errors or corruption   of data?",
    # #     "Do you use a secure channel for data in-transit(e.g TLS protocol) at all the times?",
    # #     "Are you encrypting sensitive data(including user files) at rest using strong cryptographic standards such as AES-128 OR above?",
    # #     "Critical applications should support authentication filtering based on IP address and access needs to be restricted to AMC's network.",
    # #     "In a client-server application model, the synchronization between the two should be encrypted using AMC approved encryption algorithms.",
    # #     "The SaaS provider should not:",
    # #     "To your service?",
    # #     "Do you use open standards such as Oauth to delegate authentication capabilities to your tenants?",
    # #     "Do you support identity federation standards (SAML, SPML, WS-Federation, etc.) as a means of authenticating/authorizing users?", 

    # #     "Do you have a Policy Enforcement capability (e.g., XACML) to enforce regional legal and policy constraints on user access?",
    # #     "Do you have an identity management system in place to enable both role-based and context-based access to systems including data?",
    # #     "Do you provide tenants with strong (multifactor) authentication options (digital certs, tokens, biometrics, etc.) for user access?", 

    # #     "Do you allow tenants to use third-party identity assurance services?",
    # #     "Do you provide tenants with separate environments for production and test processes?",
    # #     "Is code authorized before its installation and use, and the code configuration checked, to ensure that the authorized code operates according to a clearly defined security policy?",
    # #     "Is all unauthorized code prevented from executing?",
    # #     "Please share detailed mechanism.",
    # #     "How do you protect user authentication information?",
    # #     "How is tenant account information for the application stored?",
    # #     "Who has access to User Files?How access is controlled?",
    # #     "Please share below reports:",
    # #     "Application security(Greybox and Blackbox)",
    # #     "Secure Source Code Review",
    # #     "CA/CR of OS and Middleware/DB,Cloud Managed Services",
    # #     "VAPT of servers",
    # #     "System & Network Security",
    # #     "Who has access to productions systems from AMC side and SaaS service provider side?",
    # #     "How do personnel authenticate? How do you manage accounts?",
    # #     "How are password policies enforced?",
    # #     "Is access to the system logged?",
    # #     "How often do you patch production systems?",
    # #     "Do you quarterly review firewalls rules?",
    # #     "Others",
    # #     "Please share a list of APIs in-scope along with their appsec reports? Are APIs getting exposed to 3rd parties(including Axis AMC) via API Gatway? Please Share details.",
    # #     "At the time of termination, will you make AMC's data available to AMC upon request in compatible format"
    # # ]
        result = fetch_result_from_context(master_question_list, vector_store)
        with open(result_file_name, 'a', encoding="utf-8") as f:
            f.write(result)
            f.write('\n')
        write_to_s3_folder(bucket_name, result_folder_name, result_file_name, result)
        print("results written to s3..!")
    
    print("everything is done ..!")

main()
