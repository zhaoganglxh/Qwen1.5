import json5
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 可选的模型包括: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("/media/zhaogang/4T2-2(大语言模型)/HuggingFace/models/qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/media/zhaogang/4T2-2(大语言模型)/HuggingFace/models/qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True,bf16=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained("/media/zhaogang/4T2-2(大语言模型)/HuggingFace/models/qwen/Qwen-7B-Chat", trust_remote_code=True)

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。


history = []


class CourseDatabase:
    def __init__(self):
        self.database = {
            "大模型技术实战": {
                "课时": 211,
                "每周更新次数": 3,
                "每次更新小时": 2
            },
            "机器学习实战": {
                "课时": 230,
                "每周更新次数": 2,
                "每次更新小时": 1.5
            },
            "深度学习实战": {
                "课时": 150,
                "每周更新次数": 1,
                "每次更新小时": 3
            },
            "AI数据分析": {
                "课时": 10,
                "每周更新次数": 1,
                "每次更新小时": 1
            },
        }

    def course_query(self, course_name):
        return self.database.get(course_name, "目前没有该课程信息")


course_db = CourseDatabase()
# 查询已有课程的详细信息
course_name = "大模型技术实战"
print(course_db.course_query(course_name))

# 查询不存在课程的详细信息
course_name = "人工智能"
print(course_db.course_query(course_name))


# 定义数据库操作工具
class CourseOperations:
    def __init__(self):
        self.db = CourseDatabase()

    def add_hours_to_course(self, course_name, additional_hours):
        if course_name in self.db.database:
            self.db.database[course_name]['课时'] += int(additional_hours)
            return f"课程 {course_name}的课时已增加{additional_hours}小时。"
        else:
            return "课程不存在,无法添加课时"


course_ops = CourseOperations()
# 给某个课程增加课时

# 给某个课程增加课时
print(course_ops.add_hours_to_course("大模型技术实战", 20))

TOOLS = [
    {
        'name_for_human': '课程信息数据库',
        'name_for_model': 'CourseDatabase',
        'description_for_model': '课程信息数据库存储有各课程的详细信息,包括目前的上线课时，每周更新次数以及每次更新的小时数。通过输入课程名称，可以返回该课程的详细信息。',
        'parameters': [{
            'name': 'course_query',
            'description': '课程名称,所需查询信息的课程名称',
            'required': True,
            'schema': {
                'type': 'string'
            },
        }],
    },
    {
        'name_for_human': '课程操作工具',
        'name_for_model': 'CourseOperations',
        'description_for_model': '课程操作工具提供了对课程信息的添加操作，可以添加课程的详细信息，如每周更新次数，更新课时',
        'parameters': [{
            'name': 'add_hours_to_course',
            'description': '给指定的课程增加课时，需要课程名称和增加的课时数',
            'required': True,
            'schema': {
                'type': 'string',
                'properties': {
                    'course_name': {'type': 'string'},
                    'additional_hours': {'type': 'string'}
                },
                'required': ['course_name', 'additional_hours']
            },
        }],
    },
    # 其他工具的定义可以在这里继续添加
]

# 将一个插件的关键信息拼接成一段文本的模板
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters:{parameters}
"""

PROMPT_REACT = """Answer the following questions as best you con. You have access to the following
{tool_descs}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {query}"""

import json


def generate_action_prompt(query):
    """
    根据用户查询生成最终的动作提示字符串。
    函数内部直接引用全局变量 TOOLS, TOOL_DESC, 和 PROMPT_REACT.
    参数：
    - query: 用户的查询字符串。
    返回：
    - action_prompt: 格式化后的动作提示字符串。
    """

    tool_descs = []
    tool_names = []

    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])

    tool_descs_str = '\n\n'.join(tool_descs)
    tool_names_str = ','.join(tool_names)

    action_prompt = PROMPT_REACT.format(tool_descs=tool_descs_str, tool_names=tool_names_str, query=query)
    return action_prompt


react_stop_words = [
    tokenizer.encode('Observation:'),
    tokenizer.encode('Observation:\n'),
]


def parse_plugin_action(text: str):
    """
    解析模型的ReAct输出文本提取名称及其参数。
    参数：
    - text： 模型ReAct提示的输出文本
    返回值：
    - action_name: 要调用的动作（方法）名称。
    - action_arguments: 动作（方法）的参数。
    """
    # 查找“Action:”和“Action Input：”的最后出现位置
    action_index = text.rfind('\nAction:')
    action_input_index = text.rfind('\nAction Input:')
    observation_index = text.rfind('\nObservation:')

    # 如果文本中有“Action:”和“Action Input：”
    if 0 <= action_index < action_input_index:
        if observation_index < action_input_index:
            text = text.rstrip() + '\nObservation:'
            observation_index = text.rfind('\nObservation:')

    # 确保文本中同时存在“Action:”和“Action Input：”
    if 0 <= action_index < action_input_index < observation_index:
        # 提取“Action:”和“Action Input：”之间的文本为动作名称
        action_name = text[action_index + len('\nAction:'):action_input_index].strip()
        # 提取“Action Input：”之后的文本为动作参数
        action_arguments = text[action_input_index + len('\nAction Input:'):observation_index].strip()
        return action_name, action_arguments

    # 如果没有找到符合条件的文本，返回空字符串
    return '', ''


def execute_plugin_from_react_output(response):
    """
    根据模型的ReAct输出执行相应的插件调用，并返回调用结果。
    参数：
    - response: 模型的ReAct输出字符串。
    返回：
    - result_dict: 包括状态码和插件调用结果的字典。
    """
    # 从模型的ReAct输出中提取函数名称及函数入参
    plugin_configuration = parse_plugin_action(response)
    #first_config_line = plugin_configuration[1:][0].split('\n')[0]
    #config_parameters = json.loads(first_config_line)
    tools_msg = [TOOL for TOOL in TOOLS if plugin_configuration[0] in TOOL['name_for_model']]
    result_dict = {"status_code": 200}

    for tool_msg in tools_msg:
        for config_line in plugin_configuration[1:]:
            config_parameters = json5.loads(config_line)
            for k, v in config_parameters.items():
                if k in [param['name'] for param in tool_msg['parameters']]:
                    # 通过eval函数执行存储在字符串中的python表达式，并返回表达式计算结果。其执行过程实质上是实例化类
                    tool_instance = eval(plugin_configuration[0])()
                    # 然后通过getattr函数传递对象和字符串形式的属性或方法名来动态的访问该属性和方法h
                    tool_func = getattr(tool_instance, k)
                    # 这一步实际上执行的过程就是：course_db,course_query('大模型技术实战')
                    if isinstance(v,dict):
                        tool_result = tool_func(**v)
                    else:
                        tool_result = tool_func(v)
                    result_dict["result"] = tool_result
                    return result_dict

    result_dict["status_code"] = 404
    result_dict["result"] = "未找到匹配的插件配置"
    return result_dict


query = "先帮我查询一下大模型技术实战这个课程目前更新了多少节，今晚我直播了一节新课，请你帮我更新一下"

action_prompt = generate_action_prompt(query)

#print(action_prompt)

# 使用action_prompt生成回复
response, history = model.chat(tokenizer, action_prompt, history=None,
                               stop_words_ids=react_stop_words)
# print(response)


tool_result = execute_plugin_from_react_output(response)
# print(tool_result)

response += " " + str(tool_result)
response += " " + "\n" + query

response, history = model.chat(tokenizer, response, history=history,
                              stop_words_ids=react_stop_words)

tool_result = execute_plugin_from_react_output(response)

response += " " + str(tool_result)

response, history = model.chat(tokenizer, response, history=history,
                              stop_words_ids=react_stop_words)

print(response)