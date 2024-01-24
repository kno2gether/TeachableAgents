from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


class CustomLabelTeachability(Teachability):
    def add_to_agent(self, agent: ConversableAgent):
        """Adds teachability to the given agent."""
        self.teachable_agent = agent

        # Register a hook for processing the last message.
        agent.register_hook(hookable_method=agent.process_last_message, hook=self.process_last_message)

        # Was an llm_config passed to the constructor?
        if self.llm_config is None:
            # No. Use the agent's llm_config.
            self.llm_config = agent.llm_config
        assert self.llm_config, "Teachability requires a valid llm_config."

        # Create the analyzer agent.
        self.analyzer = TextAnalyzerAgent(llm_config=self.llm_config)

        # Append extra info to the system message.
        agent.update_system_message(
            agent.system_message
            + "\nYou've been given the special ability to remember user teachings from prior conversations."
        )
        
    def _consider_memo_storage(self, comment):
        """Decides whether to store something from one user comment in the DB."""
        memo_added = False

        # Check for a problem-solution pair.
        response = self._analyze(
            comment,
            "Does any part of the TEXT ask the agent to perform a task or solve a problem? Answer with just one word, yes or no.",
        )
        if "yes" in response.lower():
            # Can we extract advice?
            advice = self._analyze(
                comment,
                "Briefly copy any advice from the TEXT that may be useful for a similar but different task in the future. But if no advice is present, just respond with 'none'.",
            )
            if "none" not in advice.lower():
                # Yes. Extract the task.
                task = self._analyze(
                    comment,
                    "Briefly copy just the task from the TEXT, then stop. Don't solve it, and don't include any advice.",
                )
                # Generalize the task.
                general_task = self._analyze(
                    task,
                    "Summarize very briefly, in general terms, the type of task described in the TEXT. Leave out details that might not appear in a similar problem.",
                )
                # Create Sensitivity Label
                sensitivity_label = self._analyze(
                    comment,
                    "Analyze the sensitivity type of the TEXT if it contains any private, extremely personal information. if yes, then answer the label just one word, private or public and don't include any advice.",
                )
                # Add the task-advice (problem-solution) pair to the vector DB.
                if self.verbosity >= 1:
                    print(colored("\nREMEMBER THIS TASK-ADVICE PAIR", "light_yellow"))
                advice_to_save = 'sensitivity_label :' + sensitivity_label + ', advice: ' + advice
                self.memo_store.add_input_output_pair(general_task, advice_to_save)
                memo_added = True

        # Check for information to be learned.
        response = self._analyze(
            comment,
            "Does the TEXT contain information that could be committed to memory? Answer with just one word, yes or no.",
        )
        if "yes" in response.lower():
            # Yes. What question would this information answer?
            question = self._analyze(
                comment,
                "Imagine that the user forgot this information in the TEXT. How would they ask you for this information? Include no other text in your response.",
            )
            # Extract the information.
            answer = self._analyze(
                comment, "Copy the information from the TEXT that should be committed to memory. Add no explanation."
            )
            # Create Sensitivity Label
            sensitivity_label = self._analyze(
                comment,
                "Analyze the sensitivity type of the TEXT if it contains any private, extremely personal information. if yes, then answer the label just one word, private or public and don't include any advice.",
            )            
            # Add the question-answer pair to the vector DB.
            if self.verbosity >= 1:
                print(colored("\nREMEMBER THIS QUESTION-ANSWER PAIR", "light_yellow"))
            answer_to_save = 'sensitivity_label :' + sensitivity_label + ', answer: ' + answer
            self.memo_store.add_input_output_pair(question, answer_to_save)
            memo_added = True

        # Were any memos added?
        if memo_added:
            # Yes. Save them to disk.
            self.memo_store._save_memos()      