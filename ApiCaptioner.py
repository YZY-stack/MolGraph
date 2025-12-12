import openai
import json
import os
import base64
from typing import Union, List, Dict, Any, Optional


class ApiCaptioner():
    def __init__(
        self,
        key,
        base_url='https://www.chataiapi.com/v1',
        func_show=False,
        ):
        self.key = key
        self.base_url = base_url
        self.client = openai.OpenAI(api_key=self.key, base_url=self.base_url)
        if func_show:
            self.show_func()  # 调用说明函数，展示当前类的基本功能函数

    def show_func(self):
        """
        一个说明函数，展示当前类的基本功能函数
        """
        print("=" * 40)
        print("                    ApiCaptioner 类功能说明")
        print("=" * 40)
        
        functions_info = [
            {
                "name": "organize_prompt",
                "type": "实例方法",
                "description": "组织提示词，将各种格式的消息转换为API调用格式",
                "usage": "self.organize_prompt(message_list, img_detail='auto', system_prompt=None)"
            },
            {
                "name": "__call__",
                "type": "实例方法", 
                "description": "直接调用OpenAI API进行对话（支持文本和图像）",
                "usage": "captioner(messages, model='gpt-4o', max_tokens=3000, temperature=1)"
            },
            {
                "name": "batch_call_with_threading",
                "type": "实例方法",
                "description": "多线程批量调用外部函数，带进度显示",
                "usage": "self.batch_call_with_threading(func, args_list, max_workers=5)"
            },
            {
                "name": "parallel_api_calls", 
                "type": "实例方法",
                "description": "并行调用OpenAI API处理多个消息",
                "usage": "self.parallel_api_calls(messages_list, model='gpt-4o', max_workers=3)"
            },
            {
                "name": "json_write",
                "type": "类方法",
                "description": "将数据写入JSON文件",
                "usage": "ApiCaptioner.json_write(data, filename='caption.json')"
            },
            {
                "name": "jsonl_write", 
                "type": "类方法",
                "description": "将数据写入JSONL文件（每行一个JSON对象）",
                "usage": "ApiCaptioner.jsonl_write(data, filename='caption.jsonl')"
            },
            {
                "name": "image_to_base64",
                "type": "类方法",
                "description": "将图像文件转换为base64编码字符串",
                "usage": "ApiCaptioner.image_to_base64(image_path)"
            },
            {
                "name": "get_image_data_url",
                "type": "类方法", 
                "description": "获取图像的data URL格式（包含MIME类型）",
                "usage": "ApiCaptioner.get_image_data_url(image_path)"
            }
        ]
        
        print("\n📋 可用函数列表：\n")
        
        for i, func in enumerate(functions_info, 1):
            print(f"{i}. 【{func['type']}】 {func['name']}")
            print(f"   💡 功能: {func['description']}")
            print(f"   🔧 用法: {func['usage']}")
            print()
        
        print("=" * 80)
        print("🔥 主要特性:")
        print("   • 支持多模态输入（文本 + 图像）")
        print("   • 自动图像base64转换")
        print("   • 多线程批量处理")
        print("   • 进度条显示")
        print("   • 错误处理和重试机制")
        print()
        
        print("📖 快速开始:")
        print("   1. 单次调用: captioner('你好，请介绍一下人工智能')")
        print("   2. 图像分析: captioner([{'text': '描述图片'}, {'image': '/path/to/image.jpg'}])")
        print("   3. 批量处理: captioner.parallel_api_calls(messages_list)")
        print("=" * 80)

    def organize_prompt(self, message_list: Union[List[dict], List[str], str], img_detail: str='auto', system_prompt: str = None):
        """
        组织提示词，将消息列表转换为适合API调用的格式
        
        Args:
        message_list (Union[List[dict], List[str], str]): 消息列表或单个消息字符串, 若传入多模态信息，则必须使用List[dict]格式, 其中key为消息的类型，value为消息内容或者图像等文件的地址
            system_prompt (str, optional): 系统提示词，默认为None
        
        Returns:
            List[dict]: 格式化后的消息列表，适合OpenAI API调用
        """
        messages = []
        
        # 检查是否同时提供了system_prompt和message_list中的system字段
        if system_prompt and isinstance(message_list, list) and all(isinstance(item, dict) for item in message_list):
            for item in message_list:
                if any(msg_type.lower() == 'system' for msg_type in item.keys()):
                    raise ValueError("Cannot provide both 'system_prompt' parameter and 'system' field in message_list. Please use only one method to specify the system prompt.")
        
        # 添加系统提示词（如果提供）
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 处理不同类型的输入
        if isinstance(message_list, str):
            # 单个字符串消息
            messages.append({
                "role": "user",
                "content": message_list
            })
        
        elif isinstance(message_list, list) or isinstance(message_list, tuple):
            if all(isinstance(item, str) for item in message_list):
                # 字符串列表，合并为单个消息
                combined_text = "\n".join(message_list)
                messages.append({
                    "role": "user",
                    "content": combined_text
                })
            
            elif all(isinstance(item, dict) for item in message_list):
                # 多模态消息列表
                content_items = []
                
                for item in message_list:
                    for msg_type, msg_value in item.items():
                        if msg_type.lower() in ['system']:
                            # 系统提示词
                            messages.append({
                                "role": "system",
                                "content": str(msg_value)
                            })
                        
                        elif msg_type.lower() in ['text', 'string', 'str']:
                            # 文本消息
                            content_items.append({
                                "type": "text",
                                "text": str(msg_value)
                            })
                        
                        elif msg_type.lower() in ['image', 'img', 'picture', 'photo']:
                            # 图像消息 - 仅支持本地图像文件路径
                            if isinstance(msg_value, str):
                                # 检查是否为本地文件路径
                                if os.path.exists(msg_value):
                                    # 使用类内的base64转换函数
                                    data_url = self.image_to_base64(msg_value)
                                    if data_url:
                                        content_items.append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": data_url,
                                                "detail": img_detail,
                                            }
                                        })
                                    else:
                                        print(f"Warning: Failed to convert image to base64: {msg_value}")
                                else:
                                    print(f"Warning: Image file not found or not supported: {msg_value}")
                                    print("Only local image file paths are supported.")
                
                if content_items:
                    messages.append({
                        "role": "user",
                        "content": content_items
                    })
            
            else:
                # 混合类型列表，尝试转换为字符串
                combined_text = "\n".join(str(item) for item in message_list)
                messages.append({
                    "role": "user",
                    "content": combined_text
                })
        
        else:
            # 其他类型，转换为字符串
            messages.append({
                "role": "user",
                "content": str(message_list)
            })
        
        return messages
    
    def batch_call_with_threading(self, func, args_list, max_workers=5, timeout=None, show_progress=True):
        """
        使用多线程批量调用外部函数，带有进度显示
        
        Args:
            func: 要调用的外部函数
            args_list: 参数列表，每个元素是一个元组 (args, kwargs)
            max_workers: 最大工作线程数，默认为5
            timeout: 超时时间（秒），默认为None（无超时）
            show_progress: 是否显示进度条，默认为True
            
        Returns:
            List: 结果列表，包含每个函数调用的结果
            
        Examples:
            # 示例1：批量调用数学函数
            def multiply(x, y=1):
                return x * y
            
            args_list = [
                ([10], {"y": 2}),    # multiply(10, y=2) = 20
                ([20], {"y": 3}),    # multiply(20, y=3) = 60
                ([30], {"y": 4}),    # multiply(30, y=4) = 120
            ]
            
            captioner = ApiCaptioner()
            results = captioner.batch_call_with_threading(
                func=multiply,
                args_list=args_list,
                max_workers=3
            )
            # 输出: [20, 60, 120]
            
            # 示例2：批量处理文件
            def process_file(filepath, operation="read"):
                with open(filepath, 'r') as f:
                    data = f.read()
                if operation == "count":
                    return len(data)
                return data[:100]  # 返回前100个字符
            
            args_list = [
                (["file1.txt"], {"operation": "count"}),
                (["file2.txt"], {"operation": "read"}),
                (["file3.txt"], {"operation": "count"}),
            ]
            
            results = captioner.batch_call_with_threading(
                func=process_file,
                args_list=args_list,
                max_workers=2,
                timeout=30
            )
            
            # 示例3：批量网络请求
            import requests
            
            def fetch_url(url, method="GET"):
                response = requests.request(method, url)
                return response.status_code
            
            urls = ["http://example.com", "http://google.com", "http://github.com"]
            args_list = [([url], {"method": "GET"}) for url in urls]
            
            results = captioner.batch_call_with_threading(
                func=fetch_url,
                args_list=args_list,
                max_workers=3,
                show_progress=True
            )
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        import time
        import sys
        
        results = []
        futures = []
        
        if show_progress:
            print(f"Starting batch execution with {len(args_list)} tasks using {max_workers} threads...")
            print("Progress: [" + " " * 50 + "] 0%", end="")
            sys.stdout.flush()
            start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            for i, (args, kwargs) in enumerate(args_list):
                if isinstance(args, (list, tuple)):
                    future = executor.submit(func, *args, **kwargs)
                else:
                    # 如果args不是列表或元组，假设它是单个参数
                    future = executor.submit(func, args, **kwargs)
                futures.append((i, future))
            
            # 收集结果并显示进度
            completed_count = 0
            total_tasks = len(args_list)
            
            for original_index, future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append((original_index, result, None))  # (index, result, error)
                    completed_count += 1
                    
                    if show_progress:
                        # 更新进度条
                        progress = completed_count / total_tasks
                        filled_length = int(50 * progress)
                        bar = "█" * filled_length + "-" * (50 - filled_length)
                        percent = progress * 100
                        # 计算剩余时间 ETA
                        elapsed = time.time() - start_time
                        remaining = total_tasks - completed_count
                        avg = elapsed / completed_count if completed_count else 0
                        eta = avg * remaining
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                        # 打印包含 ETA 的进度条
                        print(f"\rProgress: [{bar}] {percent:.1f}% ({completed_count}/{total_tasks}) ETA: {eta_str}", end="")
                        sys.stdout.flush()
                    else:
                        print(f"Task {original_index + 1}/{total_tasks} completed successfully")
                        
                except Exception as e:
                    # 获取详细的异常信息
                    import traceback
                    error_details = f"{type(e).__name__}: {str(e)}"
                    full_traceback = traceback.format_exc()
                    
                    results.append((original_index, None, error_details))  # (index, result, error)
                    completed_count += 1
                    
                    if show_progress:
                        # 更新进度条（包含错误）
                        progress = completed_count / total_tasks
                        filled_length = int(50 * progress)
                        bar = "█" * filled_length + "-" * (50 - filled_length)
                        percent = progress * 100
                        # 计算剩余时间 ETA
                        elapsed = time.time() - start_time
                        remaining = total_tasks - completed_count
                        avg = elapsed / completed_count if completed_count else 0
                        eta = avg * remaining
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                        print(f"\rProgress: [{bar}] {percent:.1f}% ({completed_count}/{total_tasks}) ETA: {eta_str} - Error in task {original_index + 1}", end="")
                        sys.stdout.flush()
                    else:
                        print(f"Task {original_index + 1}/{total_tasks} failed with error: {error_details}")
                        print(f"Full traceback:\n{full_traceback}")
        
        if show_progress:
            print()  # 换行
        
        # 按原始索引排序结果
        results.sort(key=lambda x: x[0])
        
        # 返回结果和错误信息
        final_results = []
        error_count = 0
        for _, result, error in results:
            if error is None:
                final_results.append(result)
            else:
                final_results.append(f"Error: {error}")
                error_count += 1
        
        success_count = total_tasks - error_count
        print(f"Batch execution completed. {success_count}/{total_tasks} tasks succeeded, {error_count} failed.")
        
        # 如果有错误，显示详细的错误信息
        if error_count > 0:
            print("\n--- Error Details ---")
            for i, (_, result, error) in enumerate(results):
                if error is not None:
                    print(f"Task {i + 1} Error: {error}")
            print("--- End of Error Details ---")
        
        return final_results
    
    def parallel_api_calls(self, messages_list, model="gpt-4o", max_workers=3, show_progress=True, **api_kwargs):
        """
        并行调用OpenAI API处理多个消息
        
        Args:
            messages_list: 消息列表，每个元素是一个消息内容
            model: 使用的模型
            max_workers: 最大并行数量，建议不要太高以避免rate limit
            show_progress: 是否显示进度条，默认为True
            **api_kwargs: 其他API参数（如max_tokens, temperature等）
            
        Returns:
            List: API返回结果列表
            
        Examples:
            captioner = ApiCaptioner()
            
            # 示例1：批量文本生成
            messages_list = [
                "请描述一下春天的景色",
                "解释一下人工智能的概念", 
                "写一首关于月亮的诗"
            ]
            
            results = captioner.parallel_api_calls(
                messages_list=messages_list,
                model="gpt-4o",
                max_workers=2,
                max_tokens=1000,
                temperature=0.7
            )
            
            for i, result in enumerate(results):
                print(f"Response {i+1}: {result}")
            
            # 示例2：批量图像描述
            image_messages = [
                [{"text": "描述这张图片"}, {"image": "/path/to/image1.jpg"}],
                [{"text": "分析这张图片的颜色"}, {"image": "/path/to/image2.jpg"}],
                [{"text": "识别图片中的物体"}, {"image": "/path/to/image3.jpg"}]
            ]
            
            results = captioner.parallel_api_calls(
                messages_list=image_messages,
                max_workers=2,
                system_prompt="你是一个专业的图像分析师",
                max_tokens=500
            )
            
            # 示例3：批量翻译任务
            translation_tasks = [
                "Translate to Chinese: Hello, how are you?",
                "Translate to Chinese: Machine learning is fascinating",
                "Translate to Chinese: The weather is beautiful today"
            ]
            
            results = captioner.parallel_api_calls(
                messages_list=translation_tasks,
                model="gpt-4o",
                max_workers=3,
                system_prompt="You are a professional translator.",
                temperature=0.3
            )
        """
        # 准备参数列表
        args_list = []
        for messages in messages_list:
            args = (messages,)  # 位置参数
            kwargs = {"model": model, **api_kwargs}  # 关键字参数
            args_list.append((args, kwargs))
        
        print(f"Starting parallel API calls for {len(messages_list)} requests...")
        
        # 使用批量多线程调用
        return self.batch_call_with_threading(
            func=self.__call__,
            args_list=args_list,
            max_workers=max_workers,
            timeout=60,  # 60秒超时
            show_progress=show_progress
        )

    def __call__(self, messages, model="gpt-4o", max_tokens=3000, temperature=1, img_detail='auto', system_prompt=None):
        """
        调用OpenAI API进行对话
        
        Args:
            messages: 消息内容，可以是字符串、字符串列表或多模态消息列表
            model: 使用的模型，默认为gpt-4o
            max_tokens: 最大token数量
            temperature: 温度参数
            system_prompt: 系统提示词
            
        Returns:
            str: API返回的内容或错误信息
        """
        try:
            # 使用organize_prompt方法组织消息
            organized_messages = self.organize_prompt(messages, img_detail=img_detail, system_prompt=system_prompt)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=organized_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"

    @classmethod
    def json_write(cls, data, filename='caption.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def jsonl_write(cls, data, filename='caption.jsonl'):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

    @classmethod
    def jsonl_read(cls, filename):
        """
        从JSONL文件中读取数据

        Args:
            filename (str): JSONL文件的路径

        Returns:
            List[dict]: 包含每行JSON对象的列表
        """
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Error reading JSONL file {filename}: {e}")
        return data

    @classmethod
    def json_read(cls, filename):
        """
        从JSON文件中读取数据

        Args:
            filename (str): JSON文件的路径

        Returns:
            dict: JSON文件中的数据
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {filename}: {e}")
            return None

    @classmethod
    def image_to_base64(cls, image_path):
        """
        将图像文件读取并转换为base64格式
        
        Args:
            image_path (str): 图像文件的路径
            
        Returns:
            str: base64编码的图像数据，如果失败则返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                return None
            
            # 读取图像文件并转换为base64
            with open(image_path, 'rb') as image_file:
                # 读取二进制数据
                image_data = image_file.read()
                # 转换为base64编码
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                return base64_encoded
                
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None

    @classmethod
    def get_image_data_url(cls, image_path):
        """
        获取图像的data URL格式（包含MIME类型）
        
        Args:
            image_path (str): 图像文件的路径
            
        Returns:
            str: data URL格式的图像数据，如果失败则返回None
        """
        base64_data = cls.image_to_base64(image_path)
        if base64_data is None:
            return None
            
        # 根据文件扩展名确定MIME类型
        file_extension = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        
        mime_type = mime_types.get(file_extension, 'image/jpeg')  # 默认为jpeg
        return f"data:{mime_type};base64,{base64_data}"

