from langchain.prompts import PromptTemplate


map_prompt = """
Create a meeting summary from the following transcript discussion it should be detailed version of the summary:

"{text}"

**Meeting overview - 

** Discussion items -

** Action items - 

** Follow up tasks -
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = """
Create a meeting summary from the following transcript discussion it should be detailed version of the summary:

"{text}"

**Meeting overview - 

** Discussion items -

** Action items -

** Follow up tasks - 
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

map_prompt2 = """
Based on the transcript discussion you have reviewed, you are tasked with making a critical decision regarding:

"{text}"

Consider the following factors:

The different perspectives and arguments presented in the discussion.
The potential consequences of each decision option.
Any relevant contextual factors or constraints.

Your task is to:
Identify the key decision points and available options.
Analyze the arguments and evidence supporting each option.
Evaluate the potential risks and benefits of each option.
Make a clear and well-reasoned decision, justifying your choice based on the information available.


Remember, there may not be a single "correct" answer, but your decision should be well-informed and supported by the evidence and arguments presented in the transcript.

Output should be in structured steps - 
"""
map_prompt_template2 = PromptTemplate(template=map_prompt2, input_variables=["text"])

combine_prompt2 = """
Based on the transcript discussion you have reviewed, you are tasked with making a critical decision regarding:

"{text}"

Consider the following factors:


The different perspectives and arguments presented in the discussion.
The potential consequences of each decision option.
Any relevant contextual factors or constraints.


Your task is to:
Identify the key decision points and available options.
Analyze the arguments and evidence supporting each option.
Evaluate the potential risks and benefits of each option.
Make a clear and well-reasoned decision, justifying your choice based on the information available.


Remember, there may not be a single "correct" answer, but your decision should be well-informed and supported by the evidence and arguments presented in the transcript.

Output should be in structured steps - 
"""
combine_prompt_template2 = PromptTemplate(template=combine_prompt2, input_variables=["text"])


map_prompt3 = """
Based on the transcript discussion you have reviewed, your task is to create user story based on features identified in the transcript.

"{text}"

Consider the following Format:
As a [business user role], I want to [achieve a specific goal], so that [benefit/value to the business].

Additional to consider:
The user stories should be relevant to the business discussion.

User stories - 
Acceptance Criteria -  

For example - 

**User Story 1:**

As a data analyst, I want to access anonymized production data in a non-production environment, so that I can perform data analysis and testing without compromising data security.

**Acceptance Criteria:**

* Data anonymization scripts are developed and implemented.
* Anonymized production data is loaded into a non-production environment.
* Data analysts have access to the anonymized data for analysis and testing.
* Data security and privacy regulations are maintained.


"""

map_prompt_template3 = PromptTemplate(template=map_prompt3, input_variables=["text"])

combine_prompt3 = """
Based on the transcript discussion you have reviewed, your task is to create user story based on features identified in the transcript.

"{text}"

Consider the following Format:
As a [business user role], I want to [achieve a specific goal], so that [benefit/value to the business].

Additional to consider:
The user stories should be relevant to the business discussion.

User stories - 
Acceptance Criteria -  

For example - 

**User Story 1:**

As a data analyst, I want to access anonymized production data in a non-production environment, so that I can perform data analysis and testing without compromising data security.

**Acceptance Criteria:**

* Data anonymization scripts are developed and implemented.
* Anonymized production data is loaded into a non-production environment.
* Data analysts have access to the anonymized data for analysis and testing.
* Data security and privacy regulations are maintained.


"""

combine_prompt_template3 = PromptTemplate(template=combine_prompt3, input_variables=["text"])


map_prompt4 = """Based on the transcript discussion you have reviewed, Analyze the sentiment expressed in the discussion transcript.

"{text}"

Specific Instructions:

Identify the sentiment of each individual speaker:
- Classify the sentiment of each utterance or statement made by each speaker as positive, negative, or neutral.
- Consider the words used, the tone of voice (if available), and the context of the conversation.
Analyze the overall sentiment of the discussion:
- Determine the overall sentiment of the entire discussion, taking into account the sentiment of individual speakers and the overall flow of the conversation.
- Identify any shifts in sentiment throughout the discussion.
Identify key topics and themes that contribute to the sentiment:
- Analyze the topics and themes discussed in relation to the sentiment expressed.
- Determine if certain topics or themes are more likely to evoke positive, negative, or neutral sentiment.
Consider potential biases and limitations:
- Be aware of potential biases in the transcript, such as transcription errors or incomplete information.
- Acknowledge the limitations of sentiment analysis, such as the difficulty in capturing sarcasm or nuanced emotions.

Output:
A report summarizing the sentiment analysis, including:
- The sentiment of individual speakers
- The overall sentiment of the discussion
- Key topics and themes related to sentiment
- Potential biases and limitations
Optionally, visualizations or charts can be used to represent the sentiment analysis results."""

map_prompt_template4 = PromptTemplate(template=map_prompt4, input_variables=["text"])


combine_prompt4 = """Based on the transcript discussion you have reviewed, Analyze the sentiment expressed in the discussion transcript.

"{text}"

Specific Instructions:

Identify the sentiment of each individual speaker:
- Classify the sentiment of each utterance or statement made by each speaker as positive, negative, or neutral.
- Consider the words used, the tone of voice (if available), and the context of the conversation.
Analyze the overall sentiment of the discussion:
- Determine the overall sentiment of the entire discussion, taking into account the sentiment of individual speakers and the overall flow of the conversation.
- Identify any shifts in sentiment throughout the discussion.
Identify key topics and themes that contribute to the sentiment:
- Analyze the topics and themes discussed in relation to the sentiment expressed.
- Determine if certain topics or themes are more likely to evoke positive, negative, or neutral sentiment.
Consider potential biases and limitations:
- Be aware of potential biases in the transcript, such as transcription errors or incomplete information.
- Acknowledge the limitations of sentiment analysis, such as the difficulty in capturing sarcasm or nuanced emotions.

Output:
A report summarizing the sentiment analysis, including:
- The sentiment of individual speakers
- The overall sentiment of the discussion
- Key topics and themes related to sentiment
- Potential biases and limitations
Optionally, visualizations or charts can be used to represent the sentiment analysis results."""

combine_prompt_template4 = PromptTemplate(template=combine_prompt4, input_variables=["text"])


map_prompt5 = """Based on the transcript discussion you have reviewed and Analyze the emotions expressed in the discussion transcript.

"{text}"

Use the following emotion categories:
- Basic emotions: angry, disgust, fear, sad, happy, surprised
- Additional emotions: thankful, calm, overwhelmed, frustrated, confused, etc. (You may add or remove categories as needed.)
Analyze the overall emotional tone of the discussion:
- Determine the overall emotional tone of the entire discussion, taking into account the emotions expressed by individual speakers and the overall flow of the conversation.
- Identify any shifts in emotion throughout the discussion.
- Identify key topics and themes that contribute to the emotions expressed:
- Analyze the topics and themes discussed in relation to the emotions expressed.
- Determine if certain topics or themes are more likely to evoke specific emotions.

Output:
A report summarizing the emotion analysis, including:
The emotions expressed by individual speakers
"""


map_prompt_template5 = PromptTemplate(template=map_prompt5, input_variables=["text"])

combine_prompt5 = """Based on the transcript discussion you have reviewed and Analyze the emotions expressed in the discussion transcript.

"{text}"

Use the following emotion categories:
- Basic emotions: angry, disgust, fear, sad, happy, surprised
- Additional emotions: thankful, calm, overwhelmed, frustrated, confused, etc. (You may add or remove categories as needed.)
Analyze the overall emotional tone of the discussion:
- Determine the overall emotional tone of the entire discussion, taking into account the emotions expressed by individual speakers and the overall flow of the conversation.
- Identify any shifts in emotion throughout the discussion.
- Identify key topics and themes that contribute to the emotions expressed:
- Analyze the topics and themes discussed in relation to the emotions expressed.
- Determine if certain topics or themes are more likely to evoke specific emotions.

Output:
A report summarizing the emotion analysis, including:
The emotions expressed by individual speakers."""

combine_prompt_template5 = PromptTemplate(template=combine_prompt5, input_variables=["text"])


map_prompt6 = """Please summarize the key commitments made by each participant. For each commitment, identify the following
,Based on the transcript discussion and the provided keywords:

"{text}"

commit, assist, promise, assure, advice, guarantee, obligate, pledge, surety, confirm, definitely, inform, monitor

Who made the commitment?
What is the specific action they committed to?
Is there a deadline or timeframe associated with the commitment?
Are there any resources or support needed to fulfill the commitment?
How will progress or completion be monitored?

Additionally, please highlight any areas where commitments are unclear or require further discussion."""


map_prompt_template6 = PromptTemplate(template=map_prompt6, input_variables=["text"])



combine_prompt6 = """Please summarize the key commitments made by each participant. For each commitment, identify the following
,Based on the transcript discussion and the provided keywords:

"{text}"

commit, assist, promise, assure, advice, guarantee, obligate, pledge, surety, confirm, definitely, inform, monitor

Who made the commitment?
What is the specific action they committed to?
Is there a deadline or timeframe associated with the commitment?
Are there any resources or support needed to fulfill the commitment?
How will progress or completion be monitored?

Additionally, please highlight any areas where commitments are unclear or require further discussion."""

combine_prompt_template6 = PromptTemplate(template=combine_prompt6, input_variables=["text"])



map_prompt7 = """Please use the following questions to guide your weekly status report based on the meeting discussion transcript:

"{text}"

1. Key Decisions and Action Items:
What were the main decisions made during the meeting?
What action items were assigned, and to whom?
What are the deadlines for each action item?

2. Progress Updates:
What progress has been made on previously identified action items?
Are there any roadblocks or challenges currently being faced?
How are these challenges being addressed?

3. Discussion Highlights:
Briefly summarize the key discussion points from the meeting.
Were there any major disagreements or concerns raised?
How were these issues resolved or addressed?

4. Next Steps:
What are the next steps for the project or initiative discussed?
What are the key priorities for the upcoming week?
Are there any outstanding questions or issues that need further discussion?

5. Additional Notes:
Include any other relevant information or observations from the meeting discussion.

Please ensure your report is concise and informative, focusing on the most important takeaways from the meeting.

"""


map_prompt_template7 = PromptTemplate(template=map_prompt7, input_variables=["text"])

combine_prompt7 = """Please use the following questions to guide your weekly status report based on the meeting discussion transcript:

"{text}"

1. Key Decisions and Action Items:
What were the main decisions made during the meeting?
What action items were assigned, and to whom?
What are the deadlines for each action item?

2. Progress Updates:
What progress has been made on previously identified action items?
Are there any roadblocks or challenges currently being faced?
How are these challenges being addressed?

3. Discussion Highlights:
Briefly summarize the key discussion points from the meeting.
Were there any major disagreements or concerns raised?
How were these issues resolved or addressed?

4. Next Steps:
What are the next steps for the project or initiative discussed?
What are the key priorities for the upcoming week?
Are there any outstanding questions or issues that need further discussion?

5. Additional Notes:
Include any other relevant information or observations from the meeting discussion.

Please ensure your report is concise and informative, focusing on the most important takeaways from the meeting.

"""

combine_prompt_template7 = PromptTemplate(template=combine_prompt7, input_variables=["text"])


map_prompt8 = """Please use the following RAID framework to capture key points from the meeting discussion:

"{text}"

Risks:
What potential risks were identified during the meeting?
What is the likelihood and impact of each risk?
What mitigation strategies can be implemented to address these risks?

Actions:
What action items were assigned during the meeting?
Who is responsible for each action item?
What is the deadline for each action item?

Issues:
What current issues or challenges were discussed during the meeting?
What is the impact of each issue?
Who is responsible for resolving each issue?

Decisions:
What key decisions were made during the meeting?
Who made the decision?
What is the rationale behind each decision?

Please provide a brief description for each item within the RAID categories. This will help to track progress, identify potential roadblocks, and ensure clear communication among team members.
"""


map_prompt_template8 = PromptTemplate(template=map_prompt8, input_variables=["text"])


combine_prompt8 = """Please use the following RAID framework to capture key points from the meeting discussion:

"{text}"

Risks:
What potential risks were identified during the meeting?
What is the likelihood and impact of each risk?
What mitigation strategies can be implemented to address these risks?

Actions:
What action items were assigned during the meeting?
Who is responsible for each action item?
What is the deadline for each action item?

Issues:
What current issues or challenges were discussed during the meeting?
What is the impact of each issue?
Who is responsible for resolving each issue?

Decisions:
What key decisions were made during the meeting?
Who made the decision?
What is the rationale behind each decision?

Please provide a brief description for each item within the RAID categories. This will help to track progress, identify potential roadblocks, and ensure clear communication among team members.
"""

combine_prompt_template8 = PromptTemplate(template=combine_prompt8, input_variables=["text"])
