import autogen
from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from custom_label_teachability import CustomLabelTeachability


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json",
    file_location=".",
    filter_dict={
        "model": ["gpt-4-1106-preview", "gpt4", "gpt-4-32k"],
    },
)

teachable_agent_system_message = """
You are representing Avijit. Remember all the questions and conversation will be here related to Avijit. So, the term you always represents Avijit here. Answer user's question with both Avijit's public & private label information from the context."""

respond_teachable_agent = ConversableAgent(
    name="respond_teachable_agent",  # The name is flexible, but should not contain spaces to work in group chat.
    llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},  # Disable caching.,
    system_message=teachable_agent_system_message
)


customLabelTeachability = CustomLabelTeachability(
    verbosity=3,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=False,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

customLabelTeachability.add_to_agent(respond_teachable_agent)

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    

publicuser = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
)

text = "what is your personal email address?"
publicuser.initiate_chat(respond_teachable_agent, message=text, clear_history=True)

