Project Title:
Pete: A Mindful Companion

Authors:
Barath Chandra,Jason El Ghorayeb, Simon Pislar, Supun Madusanka

Date:
17 January 2025

Elevator Pitch:
Pete is a socially adaptive virtual robot that detects and responds to human emotions, offering personalized advice on well-being and engaging in dynamic, multi-user conversations.

Objectives (High-Level Goals)
1.	Emotion Detection Capability:
Develop an intelligent system to detect and interpret users’ emotions based on voice tone and facial expressions with a focus on accuracy and responsiveness.
2.	Multi-user Interaction:
Enable Furhat to seamlessly engage with at least two users simultaneously, identifying and responding to their unique emotional states without confusion.
3.	Adaptive Conversational Skills:
Equip Furhat with the ability to respond dynamically to users’ emotions, adjusting its tone, expressions, and conversational content to create a personalized and empathetic interaction.
4.	Wellness Advice Generation:
Design Furhat to offer actionable and practical advice regarding fitness, mental health, lifestyle, and meal planning, tailored to individual emotional states and contextual needs.
5.	Memory Integration:
Incorporate a memory feature that allows Furhat to recognize returning users, recall previous interactions, and build a long-term relationship through personalized dialogues.
6.	User-friendly Accessibility:
Create a user interface that is intuitive, easy to use, and accessible to a diverse audience, including non-technical users.
________________________________________________________________________________________
Deliverables
1.	User-Friendly Interface:
-	A clean, intuitive, and accessible user interface that allows seamless interaction with Furhat, suitable for both technical and non-technical users.
2.	Emotion Detection System:
-	A robust subsystem capable of detecting emotions in real-time from both facial expressions and voice tone.
-	Capability to handle inputs from at least two users simultaneously with minimal errors.
3.	Adaptive Robot Responses:
-	Furhat should generate appropriate and empathetic responses tailored to users' emotional states, ensuring engaging and context-sensitive interactions.
4.	Wellness and Lifestyle Advice:
-	A built-in functionality where Furhat provides advice related to fitness, mental health, lifestyle, and meal planning based on user emotional and conversational contexts.
5.	Multi-user Support:
-	Ensure Furhat can distinguish between and interact with two or more users during the same session, dynamically adapting to their respective inputs.
6.	Memory Functionality:
-	Implement a memory system where Furhat can recognize returning users, recall prior interactions, and maintain a personalized dialogue history.
7.	Real-Time Feedback Loop:
-	A system that integrates real-time emotional input (via webcam and audio) with adaptive conversation, demonstrating smooth and timely responses.
8.	Comprehensive Documentation and Presentation:
Include written documentation describing:
- Technical details of the User Perception and Interaction subsystems.
-	How the system detects emotions and generates responses.
-	Ethical considerations for user memory and privacy.
-	Deliver a final presentation showcasing Furhat’s functionality, user interaction scenarios, and real-world applications.
______________________________________________________________________________________
Success Metrics
1.	Emotion Detection:
-	Achieve at least 85% accuracy in detecting emotions based on facial expressions and voice tone during testing.
-	Validate successful detection in low-light and noisy environments.
2.	Multi-user Functionality:
-	Demonstrate Furhat’s ability to distinguish and process emotional states from at least two simultaneous users without mixing inputs.
3.	Adaptability:
-	Ensure Furhat adjusts its responses to user emotions in at least 90% of test cases during prototype trials.
4.	Advice Effectiveness:
-	Conduct user surveys to confirm that >80% of users find the wellness advice relevant and helpful.
5.	Memory Implementation:
-	Validate Furhat’s ability to recognize and recall at least 90% of returning users and adapt conversations accordingly.
6.	UI Accessibility:
-	Gather feedback from a user group to achieve at least 90% satisfaction with the UI design in terms of usability and simplicity.
_____________________________________________________________________________________
Potential Issues
1.	Emotion Detection:
-	Difficulty in accurately detecting complex emotions or ambiguous expressions.
-	Misalignment of emotion data between facial expressions and voice tone.
2.	Multi-user Interaction:
-	Challenges in distinguishing and prioritizing inputs from multiple users, especially in noisy settings.
3.	Memory Feature:
-	Potential for data storage/privacy concerns related to remembering user interactions.
-	Managing scalability as the memory database grows with repeated interactions.
4.	Real-time Processing:
-	Ensuring Furhat processes facial, audio, and conversational data in real time without delays, especially when handling multiple users.
5.	Advice Personalization:
-	Risk of generic or irrelevant advice if the system fails to adapt to diverse emotional contexts.
6.	Usability:
-	Balancing simplicity for non-technical users with advanced options for detailed customization.
-	Risk of overwhelming users with too many features or unclear navigation.
_____________________________________________________________________________________

Project Breakdown
________________________________________
1. Initial Planning & Research
Deadline: [Fri 29 Nov]
Responsible Members: Simon Pislar, Supun Madusanka, Barath Chandra, Jason El Ghorayeb
-	Finalize project scope, goals, and deliverables.
- Research emotion detection techniques (facial expressions, voice tone).
-	Review MultiEmoVA and DiffusionFER datasets and choose appropriate one for the emotion detection system.
-	Research tools and frameworks needed for Furhat's interaction subsystem (Py-Feat, machine learning models, etc.).
________________________________________
2. User Interface Design
Deadline: [Mon 9 Dec]
Responsible Members: Simon Pislar, Jason El Ghorayeb
-	Design the basic UI layout for Furhat’s interaction, focusing on accessibility and simplicity.
-	Implement initial front-end for interaction (e.g., buttons, user input fields).
-	Ensure compatibility with Furhat's hardware and speech synthesis system.
________________________________________
3. Emotion Detection Subsystem Development
Deadline: [Wed 18 Dec]
Responsible Members: Supun Madusanka, Barath Chandra
-	Implement emotion detection using facial expression analysis (Py-Feat).
-	Train machine learning models using the chosen dataset (MultiEmoVA or DiffusionFER).
-	Test and fine-tune emotion recognition accuracy with real-time webcam input.
________________________________________
4. Multi-user Interaction Testing & Enhancement
Deadline: [Wed 18 Dec]
Responsible Members: Simon Pislar, Jason El Ghorayeb
-	Develop the logic for detecting and processing emotional states from two or more users simultaneously.
-	Test the system with multiple users, ensuring correct detection and response.
________________________________________
5. Adaptive Behavior and Conversation Design
Deadline: [Wed 18 Dec]
Responsible Members: Supun Madusanka, Barath Chandra
-	Design and implement rule-based responses that adjust based on detected emotions.
-	Develop conversational strategies for offering wellness advice (fitness, mental health, etc.).
-	Integrate emotional context into the conversation flow to ensure relevancy.
________________________________________
6. Memory System Development
Deadline: [Wed Jan 15]
Responsible Members: Simon Pislar, Jason El Ghorayeb
-	Design and implement a memory function that tracks and recalls previous user interactions.
-	Ensure privacy and ethical considerations regarding user data storage.
-	Test the memory feature for accurate user recognition and dialogue adaptation.
________________________________________
7. System Integration & Testing
Deadline: [Wed Jan 15]
Responsible Members: All Members
-	Integrate emotion detection subsystem, adaptive behavior system, and memory functionality.
-	Conduct comprehensive system testing to ensure smooth user interaction.
-	Perform real-time testing of the system with multiple users.
________________________________________
8. Documentation & Presentation Preparation
Deadline: [Wed Jan 15]
Responsible Members: All Members
-	Prepare a technical report documenting the design, implementation, and testing processes.
-	Create a final presentation showcasing the system, highlighting its capabilities, real-world use cases, and challenges.
-	Include privacy and ethical considerations in the report.
________________________________________
9. Final Presentation & Report Submission
Deadline: [Wed Jan 15]
Responsible Members: All Members
-	Present the final project demo and findings to the course instructors or peers.
-	Submit the final report detailing all subsystems, design decisions, and challenges encountered.

