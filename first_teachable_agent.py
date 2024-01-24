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

print(config_list[0]["model"])

teachable_agent = ConversableAgent(
    name="teachable_agent",  # The name is flexible, but should not contain spaces to work in group chat.
    llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},  # Disable caching.
)

# Default: Instantiate the Teachability capability. Its parameters are all optional.
# teachability = Teachability(
#     verbosity=3,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
#     reset_db=False,
#     path_to_db_dir="./tmp/notebook/teachability_db",
#     recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
# )

customLabelTeachability = CustomLabelTeachability(
    verbosity=3,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=False,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

# # Default: Now add the Teachability capability to the agent.
customLabelTeachability.add_to_agent(teachable_agent)

# teachability.add_to_agent(teachable_agent)

# For this test, create a user proxy agent as usual.
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# This function will return once the user types 'exit'.
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")