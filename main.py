# -*- coding: utf-8 -*-
"""
Created on Sat May  3 22:23:10 2025

@author: zidan
"""

import json
import openai
import pprint
import re


from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt

import os
import itertools



from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)


from typing import List, Dict

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict, Counter

from vllm import LLM
from vllm.sampling_params import SamplingParams, StructuredOutputsParams


client = OpenAI(api_key= "** put the key here **") # TODO: put the key into an environment variable

#llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct") # for VLLM
#llm = LLM(model="google/gemma-4-31B-it") # for VLLM
llm = LLM(
    model="google/gemma-4-31B-it",
    tensor_parallel_size=2,
    max_model_len=16384,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 0, "audio": 0},
)

author = 'gold'
context_sent_size = 10
N = 3000
category_list = ["agentive", "low_agentive", "passive"]

label_map = {
    "passive": 0,
    "agentive": 1,
    "low_agentive": 2
}

reverse_label_map = {v: k for k, v in label_map.items()}

folder_data_path = './data'

BASE_LOG_DIR = './logs'
CHAR_LISTS_LOG_DIR = os.path.join(BASE_LOG_DIR, 'char_lists_logs') # prompt logs for the participants list
LABELS_LOG_DIR = os.path.join(BASE_LOG_DIR, 'labels_logs') # prompt logs for the labels list

pretty_log_name = 'pretty_log.json'
machine_log_name = 'machine_log.jsonl'
full_log_name = 'full_text_log.txt'

aux_prompt_data_file = 'auxiliary_prompt_data.json'





# для промптов о списке участников во фразе:
class CharListResponse(BaseModel):
    participants: List[str] = Field(default_factory=list)
    

# для промптов о лейбле персонажа во фразе:
class AgentiveClass(Enum):
     AGENTIVE = "agentive"
     LOW_AGENTIVE = "low_agentive"
     PASSIVE = "passive"
     
# тоже для второго
class AgentiveResponse(BaseModel):
     #explanation: str
     prediction: AgentiveClass
     class Config:
         use_enum_values = True


#combining annotation tags by authors (converting into a more convenient format)
#--- full_annotations - the resulting data frame of updated format
#--- output.json - file, where formated data is being saved
def combining_annotations_by_authors(data):
    full_annotations = {}
    for author in data["participations"]:
        full_annotations[author] = {
            "mentions": data.get("mentions", {}).get(author, {}),
            "participations": data["participations"][author]
        }
    full_annotations["title"] = data["title"]
    full_annotations["text"] = data["text"]
    full_annotations["sentences"] = [sent for sent in data["sentences"] if sent] #пустые исключаем
    full_annotations["sentences_with_ids"] = [
        {'id': sent_id, 'span': sent_span, 'text': data['text'][sent_span[0] : sent_span[1]]} 
        for sent_id, sent_span in enumerate(full_annotations["sentences"], 0)
        if sent_span
    ]
    
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(full_annotations, f, ensure_ascii=False, indent=2)     
        
    return full_annotations



def intersect_spans(span1, span2):
    # span1 = [a, b], span2 = [c, d]
    a, b = span1
    c, d = span2

    # Находим максимум из начальных точек и минимум из конечных
    start = max(a, c)
    end = min(b, d)

    # Если отрезки пересекаются, возвращаем их пересечение
    if start <= end:
        return [start, end]
    else:
        # Нет пересечения, возвращаем None или пустой список, как тебе удобней
        return []
    
    

#printing text for particular list of spans:
def text_by_spans(spans, full_annotations):
    text = []
    for a, b in spans:
        text.append(full_annotations["text"][a:b])
    text = '...'.join(text)
    text.replace('\r', '').replace('\n', ' ')
    print(text)
    return text




#-- text - цельный текстовый кусок
#-- spans_of_marking - список числовых пар [a, b] где надо звёзды поставить
def mark_text_with_asterisks(text, spans_of_marking):
    old_text = text
    # сортируем интервалы по убыванию, чтобы не сбивать индексы - это главный лайфхак
    spans_of_marking = sorted(spans_of_marking, reverse=True)
    for start, end in spans_of_marking:
        # вставляем закрывающие звёздочки после end (end включительно, поэтому +1)
        text = text[:end] + '**' + text[end:]
        # вставляем открывающие звёздочки перед start
        text = text[:start] + '**' + text[start:]
    return text
    




#!! полагаем, что спаны из spans могут цеплять несколько предложений
#--getting sentences for given spans
def get_sentences_overlapping_spans(full_annotations, spans, min_overlap = 0):
    # Найдём минимальный старт и максимальный конец всех spans
    min_part_start = min(span[0] for span in spans)
    max_part_end = max(span[1] for span in spans)
    
    # Проверка на пересечение
    sentence_list = []
    #intersected = False
    for sentence in full_annotations["sentences"]:
        #start and end of the current sentence:
        start_of_sentence = sentence[0]
        end_of_sentence = sentence[1]

        # Если начало предложения стоит после конца самого правого спана — прерываем цикл
        if start_of_sentence > max_part_end:
            break  # упорядочены — дальше пересечений не будет

        # Если конец предложения стоит до начала самого левого спана - сразу идём дальше 
        if end_of_sentence < min_part_start:
            continue

        #for a, b in mention_spans: #a, b - пара числе (т.е. спан). И так для каждого спана из mention_spans
        for c, d in spans:
            overlap_start = max(start_of_sentence, c)
            overlap_end = min(end_of_sentence, d)
            if overlap_end - overlap_start >= min_overlap: #значит пересеклись - возвращаем тру и предложение
                #sentence_list.append(full_annotations["text"][start_of_sentence:end_of_sentence])
                sentence_list.append([start_of_sentence, end_of_sentence]) #возвращаем спаны предложений, а не сами предложения
                break # current sentence has been already added, we have to go to the next one
                #return 1, full_annotations["text"][start_of_sentence:end_of_sentence] 
    return sentence_list

    


#getting mention span associated with particular participation spans and character:
#--it could be more than 1 mention span!
def get_mention_spans(participation_spans, full_annotations, author, pers, min_overlap = 2):
    
    result_spans = []
        
    # Найдём минимальный старт и максимальный конец всех participation spans
    min_part_start = min(span[0] for span in participation_spans)
    max_part_end = max(span[1] for span in participation_spans) 

    for mention in full_annotations[author]["mentions"].get(pers, []):
        mention_spans = mention["spans"]

        # Если все mention_spans начинаются после конца participation_spans — прерываем цикл
        if all(start > max_part_end for (start, _) in mention_spans):
            break  # упорядочены — дальше пересечений не будет

        # Если все mention_spans заканчиваются до начала participation — пропускаем
        if all(end < min_part_start for (_, end) in mention_spans):
            continue

        for a, b in mention_spans: #a, b - пара чисел (т.е. спан). И так для каждого спана из mention_spans
            for c, d in participation_spans:
                if b < c:
                    break  # так как упорядочены, дальше смысла проверять нет
                overlap_start = max(a, c)
                overlap_end = min(b, d)
                if overlap_end - overlap_start >= min_overlap: #значит пересеклись
                    #result_spans.append([a, b]) - было так. Типа возвращали мэншэн целиком.
                    result_spans.append([overlap_start, overlap_end]) # возвращаем именно само пересечение, а не целиком мэншэн
                    break #чтобы избежать многократного добавления одного и того жe спана [a, b]
                    
    return result_spans #if there is no intersection - returning an empty list.
    



# #collecting contexts and chars for further prompts preparation
# #--- N - number of prompts (process firsn N participation objects from our JSON)
# #--- every object from contexts_and_chars contains both context (of the spans) and chars, 
# #--- assosiated with particular participation object.
# #--- returning prepared data for N prompts 
# def prepare_prompt_data(full_annotations, N, author):
#     participations = full_annotations[author]["participations"][0:N]
#     contexts_and_chars = []
#     count_prompts = 0
    
#     for elem in participations:
#         tmp_spans = elem["spans"] #all the spans of current participations
#         tmp_chars = []            #all chars of current participation
        
#         #getting chars of current participation object:
#         for category in ["agentive", "low_agentive", "passive"]:
#             if elem[category]:
#                 for char in elem[category]:
#                     tmp_chars.append((char, category)) #категория здесь как "правильный лейбл". Потом для подсчёта точности пригодится
#                     #tmp_chars.extend(elem[category]) # добавит сразу всех участников данного типа, если они есть. Extend - чтобы всегда был список строк, и не было вложенной шляпы типа [['Jeronimo']]
        
   
#         for char, category in tmp_chars:              
#             #TODO: предусмотреть, что результат может быть пустым списком: 
#             #getting mention spans of current char in the current participation spans
#             char_mention_spans = get_mention_spans(tmp_spans, full_annotations, author, char) 
                    
#             #TODO: вынести в функцию отдельную
#             #new wrapping of phrase:
#             processed_phrase = [] # будет СПИСОК кусков уже полностью готовой фразы, который я объединю сепаратором "..."
#             for p_span in tmp_spans: # последовательно обрабатываем каждый кусок phrase
#                 marking_spans = []  #спаны, которые надо выделить звёздами в текущем куске фразы
#                 p_start, p_end = p_span # спан куска фразы
#                 for m_span in char_mention_spans: # сравниваем кусок фразы со всеми кусками мэншенов
#                     overlap_span = intersect_spans(p_span, m_span)
#                     if (overlap_span):
#                         overlap_span = [v - p_start for v in overlap_span] #нормализуем отрезок: полагаем, что ноль - это начало куска фразы
#                         marking_spans.append(overlap_span)
#                 #наконец передаём кусок фразы (в текстовом виде) и пронормированные спаны для звёзд в функцию:
#                 part = mark_text_with_asterisks(full_annotations["text"][p_start:p_end], marking_spans)
#                 processed_phrase.append(part)
#             processed_phrase = ' ... '.join(processed_phrase)
#             processed_phrase = processed_phrase.replace('\r', '').replace('\n', ' ')    
                
            
#             #new wrapping of context:
#             context_spans = get_sentences_overlapping_spans(full_annotations, tmp_spans)
#             processed_context = []
#             for c_span in context_spans:
#                 marking_spans = []
#                 c_start, c_end = c_span
#                 for m_span in char_mention_spans: 
#                     overlap_span = intersect_spans(c_span, m_span)
#                     if (overlap_span):
#                         overlap_span = [v - c_start for v in overlap_span] #нормализуем отрезок: полагаем, что ноль - это начало куска фразы
#                         marking_spans.append(overlap_span) 
#                 #наконец передаём кусок контекста (предложение в текстовом виде) и пронормированные спаны для звёзд в функцию:
#                 part = mark_text_with_asterisks(full_annotations["text"][c_start:c_end], marking_spans)
#                 processed_context.append(part)                                                 
#             processed_context = ' ... '.join(processed_context)
#             processed_context = processed_context.replace('\r', '').replace('\n', ' ')     
            
                
#             #adding data for a prompt:
#             contexts_and_chars.append({"char": char, "context": processed_context, "phrase": processed_phrase, "golden_label": category})
#             count_prompts += 1
                   
#     #downloading in the file:
#     with open("context_and_chars.json", "w", encoding="utf-8") as f:
#         json.dump(contexts_and_chars, f, ensure_ascii=False, indent=2)    
#     print(count_prompts)
    
#     return contexts_and_chars




# a new version of the function
def prepare_prompt_data():
    
    json_files = [f for f in os.listdir(folder_data_path) if f.endswith('.json')]
    
    prompt_data = []
    
    for file in json_files:
        with open(f'data/{file}', "r", encoding="utf-8") as f:
            data = json.load(f)
        full_annotations = combining_annotations_by_authors(data)
        full_char_list = get_complete_char_list(full_annotations, author) #getting a complete list of chars of the current file
        text = full_annotations['text']
        title = full_annotations['title']
        
        valid_iter = (  # уважаемый итератор. Он не создаёт список, а работает как поток, возвращая элемы по мере запроса.
            p for p in full_annotations[author]['participations'] 
            if is_valid_participation(p, full_annotations, author) 
        )
        
        selected_phrases = list(itertools.islice(valid_iter, N)) # берёт N элементов из потока итератора (то есть его срез - slice)
    
        print(
            f'{full_annotations["title"]}: '
            f'There are {len(selected_phrases)} from {N} valid participations, ' 
            f'according to a context-window rule.\n'
        ) 
        
        cur_data_instances = 0
        
        for idx, elem in enumerate(selected_phrases, cur_data_instances):
            phrase_spans = sorted(set(tuple(span) for span in elem['spans'])) # из-за приколов с хэшируемыми объектами. Лист - не хешируемый, а тупл - да. Поэтому привели к виду тупл.
            phrase_spans = [
                list(span) 
                for span in phrase_spans # обратно в список преобразуем
            ] 
            phrase_text = [ text[start:end] for start, end in phrase_spans ]
            phrase_text = ' ... '.join(phrase_text)
            
            phrase_sentences_spans = get_sentences_overlapping_spans(full_annotations, phrase_spans) # get sentences that the phrase intersects with
            phrase_sentences_ids = [ # number(s) of sentence(s) intersacted by the phrase
                tmp['id']
                for tmp in full_annotations['sentences_with_ids']
                if tmp['span'] in phrase_sentences_spans
            ]
            phrase_sentences_ids.sort()
            
            min_id = phrase_sentences_ids[0]
            max_id = phrase_sentences_ids[-1]
            
            context_with_phrase_text = [
                sent['text'] 
                for sent in full_annotations['sentences_with_ids'][min_id - context_sent_size : max_id + context_sent_size]
                
            ]
            
            context_with_phrase_spans = [
                sent['span']
                for sent in full_annotations['sentences_with_ids'][min_id - context_sent_size : max_id + context_sent_size]
            ]
            
            # A set of gold participants to evaluate the first prompt
            set_of_golden_chars = { 
                char 
                for category in category_list
                for char in elem[category]  
            }
            
            # A dictionary <label>: <chars> to evaluate the second prompt    
            golden_labels_with_chars = {
                category: list(elem[category]) # list(..) надо чтобы shallow copy была, т.е. новый объект создался, а не просто ссылка на тот же объект подставилась!
                for category in category_list
            }
            
            prompt_data.append(
                {   
                    "idx": idx,
                    "doc_title": title,
                    "phrase_spans": phrase_spans,
                    "phrase_text": phrase_text,
                    "context_spans": context_with_phrase_spans,
                    "context_text": ' '.join(context_with_phrase_text),
                    "all_text_chars": full_char_list,
                    "golden_chars": sorted(set_of_golden_chars), # важно! Изначально там set лежит, а его нельзя сюреализовать, поэтому ошибка была. А sorted - выдаёт уже LIST, да ещё и сортированный.
                    "golden_labels_with_chars": golden_labels_with_chars
                    
                }
            )
            
            cur_data_instances += 1
    
    
    prompt_data_log_path = os.path.join(BASE_LOG_DIR, aux_prompt_data_file)
    print(f'Total data instances: {cur_data_instances}\n')
    with open(prompt_data_log_path, "a", encoding='UTF-8') as f:
        json.dump(prompt_data, f, ensure_ascii=False, indent=4)
    
    return prompt_data

        
        
    
    
    
    
 # =====================================ЧАСТЬ СТАРОЙ ВЕРСИИ ФУНКЦИИ:===========================   
    # contexts_and_chars = []
    # count_prompts = 0
    
    # for elem in participations:
    #     tmp_spans = elem["spans"] #all the spans of current participations
    #     tmp_chars = []            #all chars of current participation
        
    #     #getting chars of current participation object:
    #     for category in ["agentive", "low_agentive", "passive"]:
    #         if elem[category]:
    #             for char in elem[category]:
    #                 tmp_chars.append((char, category)) #категория здесь как "правильный лейбл". Потом для подсчёта точности пригодится
    #                 #tmp_chars.extend(elem[category]) # добавит сразу всех участников данного типа, если они есть. Extend - чтобы всегда был список строк, и не было вложенной шляпы типа [['Jeronimo']]
        
   
    #     for char, category in tmp_chars:              
    #         #TODO: предусмотреть, что результат может быть пустым списком: 
    #         #getting mention spans of current char in the current participation spans
    #         char_mention_spans = get_mention_spans(tmp_spans, full_annotations, author, char) 
                    
    #         #TODO: вынести в функцию отдельную
    #         #new wrapping of phrase:
    #         processed_phrase = [] # будет СПИСОК кусков уже полностью готовой фразы, который я объединю сепаратором "..."
    #         for p_span in tmp_spans: # последовательно обрабатываем каждый кусок phrase
    #             marking_spans = []  #спаны, которые надо выделить звёздами в текущем куске фразы
    #             p_start, p_end = p_span # спан куска фразы
    #             for m_span in char_mention_spans: # сравниваем кусок фразы со всеми кусками мэншенов
    #                 overlap_span = intersect_spans(p_span, m_span)
    #                 if (overlap_span):
    #                     overlap_span = [v - p_start for v in overlap_span] #нормализуем отрезок: полагаем, что ноль - это начало куска фразы
    #                     marking_spans.append(overlap_span)
    #             #наконец передаём кусок фразы (в текстовом виде) и пронормированные спаны для звёзд в функцию:
    #             part = mark_text_with_asterisks(full_annotations["text"][p_start:p_end], marking_spans)
    #             processed_phrase.append(part)
    #         processed_phrase = ' ... '.join(processed_phrase)
    #         processed_phrase = processed_phrase.replace('\r', '').replace('\n', ' ')    
                
            
    #         #new wrapping of context:
    #         context_spans = get_sentences_overlapping_spans(full_annotations, tmp_spans)
    #         processed_context = []
    #         for c_span in context_spans:
    #             marking_spans = []
    #             c_start, c_end = c_span
    #             for m_span in char_mention_spans: 
    #                 overlap_span = intersect_spans(c_span, m_span)
    #                 if (overlap_span):
    #                     overlap_span = [v - c_start for v in overlap_span] #нормализуем отрезок: полагаем, что ноль - это начало куска фразы
    #                     marking_spans.append(overlap_span) 
    #             #наконец передаём кусок контекста (предложение в текстовом виде) и пронормированные спаны для звёзд в функцию:
    #             part = mark_text_with_asterisks(full_annotations["text"][c_start:c_end], marking_spans)
    #             processed_context.append(part)                                                 
    #         processed_context = ' ... '.join(processed_context)
    #         processed_context = processed_context.replace('\r', '').replace('\n', ' ')     
            
                
    #         #adding data for a prompt:
    #         contexts_and_chars.append({"char": char, "context": processed_context, "phrase": processed_phrase, "golden_label": category})
    #         count_prompts += 1
                   
    # #downloading in the file:
    # with open("context_and_chars.json", "w", encoding="utf-8") as f:
    #     json.dump(contexts_and_chars, f, ensure_ascii=False, indent=2)    
    # print(count_prompts)
    
    # return contexts_and_chars




# checking whether the participation "p" is completely located after first "CONTEXT_SIZE" sentences and before last "CONTEXT_SIZE" sentences
def  is_valid_participation(p, full_annotations, author):
    num_sentences = len(full_annotations['sentences'])
    left_border = full_annotations['sentences'][context_sent_size - 1][1] # ending of the last "excluded" sentence in the begining of the text
    right_border = full_annotations['sentences'][num_sentences - context_sent_size][0] #begining of the first "excluded" sentence in the ending of the text
    return all(
        span[0] > left_border and span[1] < right_border
        for span in p['spans']
    )

        

#getting a complete list of all the characters of the corresponding file:
def get_complete_char_list(full_annotations, author):
    char_list = [char for char in full_annotations[author]["mentions"]]
    #print(f'{full_annotations["title"]} list of chars: {char_list}')
    return char_list



def Send_Char_List_Prompt(instance, pretty_log_file, machine_log_file, text_log_file):
    
    template_messages = [
             {
                 "role": "system", 
                 "content": "You are a linguistic annotator specializing in agentivity classification in literary texts."
             },
             
             {
                 "role": "user", 
                 "content": """

Full list of characters (use ONLY these names, exactly as written):
    
{complete_char_list}


Context excerpt:
    
{context_text} 


Target phrase:
    
"{phrase_text}"


DEFINITIONS:
    
CHARACTERS AND CHARACTER ACTIONS:

All entities present in a text are considered characters if they are explicitly attributed the ability to act, communicate, or at least to think, feel and perceive in the text itself. 
These abilities are considered simultaneously as character actions. Character actions include intentional actions and non-intentional behaviors, speech or communication acts, but also internal events or feelings, and thus actions that are performed as well as imagined or unrealized actions.


SPECIAL CHARACTER VALUES:

Besides character names, the value <crowd> may be used to capture collective entities. If an identifiable character participates in the actions of a group of characters not named individually in the text (a collective entity), the corresponding text unit should be assigned two values: once with the character’s name and once with the value <crowd>.

The value <background_character> is used for characters who play a subordinate role in the text. They are part of the narrative world but do not fulfill any essential function for the plot or the development of the main characters. They are usually not referred to by a proper name in the text and are instead described by generic terms (e.g., "the woman", "the traveler", "the saleswoman") and mentioned only once or a few times.


TASK:
    
Identify which characters from the full list of characters are involved in the character actions of the EVENT described by the target phrase.
Use the context excerpt only for disambiguation and reference resolution. Identify characters based strictly on the character actions expressed in the target phrase.


Return ONLY valid JSON:
{{ 
     "participants": ["Name1", "Name2"] 
}}

If none participate:
{{ 
     "participants": [] 
}}

""".strip()
            }
    ]
 
    #filling template for the current prompt with the real data:
    complete_msg = [] # make it empty
    complete_msg.append(template_messages[0])
    complete_msg.append(
        {
            "role": "user",
            "content":
                template_messages[1]["content"].format( # для обычного промпта без прикреплённого пдф
                complete_char_list = instance["all_text_chars"],
                context_text = instance["context_text"],
                phrase_text = instance["phrase_text"]
                )
        } 
    )

    # prepearing params for response:
    schema = CharListResponse.model_json_schema() # getting python dict for the response schema


    sampling_params = SamplingParams(
        temperature = 0,
        max_tokens = 1024,
        structured_outputs = StructuredOutputsParams(json = schema)
    )


    MAX_RETRIES = 3 
    all_chars_set = set(instance["all_text_chars"])
    attempt = 0
    cleaned_response = None
    while attempt < MAX_RETRIES:
        try:     
            #sending the prompt to the model:  
            # response = client.beta.chat.completions.parse(
            #     #model="gpt-5-mini", # TODO: check the name of the model
            #     model = "gpt-5.2",
            #     messages = complete_msg,
            #     response_format = CharListResponse
            # )

            response = llm.chat(
                [complete_msg],
                sampling_params
            )
            
            
        except Exception as e:
            if "context_length_exceeded" in str(e):
                attempt += 1
                continue
            else: 
                print(f"[X] Unexpected error: {e}")
                break
        
        #converting the response to the json format:
        #cleaned_response = response.choices[0].message.parsed.model_dump()
        text_response = response[0].outputs[0].text
        schema_object_response = CharListResponse.model_validate_json(text_response) # trying to convert text respond to python dictionary. Output - OBJECT of CharListResponse! 
        cleaned_response = schema_object_response.model_dump() # cleaned response, which is python dict.
        
        if cleaned_response is None:
            attempt += 1
            continue
        invalid = set(cleaned_response['participants']) - all_chars_set
        if invalid:
            print(f'[X] Wrong character name: {invalid}')
            attempt += 1
            continue
        break
    
    
    success = True
    if cleaned_response is None:
        success = False
    else: 
        invalid = set(cleaned_response["participants"]) - all_chars_set
        if invalid:
            success = False          
    if not success:
        print(f'[X] Failed after {MAX_RETRIES} attemts')
        return False, []
    
    
    #--------------------Логированипе в pretty и в machine форматах------------
    log_entry = {
        'idx': instance['idx'],
        'doc_title': instance['doc_title'],
        'phrase_spans': instance['phrase_spans'],
        'phrase_text': instance['phrase_text'],
        'context_spans': instance['context_spans'],
        'context_text': instance['context_text'],
        'all_text_chars': instance['all_text_chars'],
        'gold_char_list': instance['golden_chars'],
        "golden_labels_with_chars": instance["golden_labels_with_chars"],
        'predicted_char_list': cleaned_response["participants"],    
    }
    
    json.dump(log_entry, pretty_log_file, ensure_ascii = False, indent = 4) # Логированипе в красивый json файл для ручного просмотра:
    
    json.dump(log_entry, machine_log_file, ensure_ascii = False)  # Логированипе в jsonlines файл для дальнейшего автоматического разбора 
    machine_log_file.write("\n") # обязательно добавлять \n чтобы соблюсти правила jsonlines!
    
    #----------------------Логирование в текстовый файл более подробно:-------------------
    text_log_file.write("=" * 80 + "\n")
    text_log_file.write(f"[ID]     {instance['idx']}\n")
    text_log_file.write(f"[PHRASE SPANS]        {instance['phrase_spans']}\n")
    text_log_file.write(f"[PHRASE TEXT]        {instance['phrase_text']}\n")
    text_log_file.write(f"[CONTEXT SPANS]        {instance['context_spans']}\n")
    text_log_file.write(f"[CONTEXT TEXT]        {instance['context_text']}\n")
    text_log_file.write(f"[GOLD_CHAR_LIST]    {instance['golden_chars']}\n")
    text_log_file.write(f"[PREDICTION]    {cleaned_response['participants']}\n")
    text_log_file.write("\n")
    
    text_log_file.write("[PROMPT]\n")
    text_log_file.write("-" * 80 + "\n")
    text_log_file.write(complete_msg[1]['content'] + "\n")
    text_log_file.write("-" * 80 + "\n\n")
    
    #text_log_file.write("[EXPLANATION]\n")
    #text_log_file.write(cleaned_response["explanation"] + "\n")
    text_log_file.write("=" * 80 + "\n\n")
    
    return True, log_entry



def Send_Labels_Prompt(instance, doc_annotations, pretty_log_file, machine_log_file, text_log_file):
    
    MAX_RETRIES = 3
    template_messages = [
             {
                 "role": "system", 
                 "content": "You are a linguistic annotator specializing in agentivity classification in literary texts."
             },
             
             {
                 
                 "role": "user", 
                 "content": """

Character:

{character}


Context excerpt:
     
{context_text} 


Target phrase:
     
"{phrase_text}"


DEFINITIONS:
     
CHARACTERS AND CHARACTER ACTIONS:

All entities present in a text are considered characters if they are explicitly attributed the ability to act, communicate, or at least to think, feel and perceive in the text itself. 
These abilities are considered simultaneously as character actions. Character actions include intentional actions and non-intentional behaviors, speech or communication acts, but also internal events or feelings, and thus actions that are performed as well as imagined or unrealized actions.


SPECIAL CHARACTER VALUES:

Besides character names, the value <crowd> may be used to capture collective entities. If an identifiable character participates in the actions of a group of characters not named individually in the text (a collective entity), the corresponding text unit is assigned two values: once with the character’s name and once with the value <crowd>.

The value <background_character> is used for characters who play a subordinate role in the text. They are part of the narrative world but do not fulfill any essential function for the plot or the development of the main characters. They are usually not referred to by a proper name in the text and are instead described by generic terms (e.g., "the woman", "the traveler", "the saleswoman") and mentioned only once or a few times.


INSTRUCTIONS HOW TO CHOSE A LABEL:

{labels_description}
 
 
TASK:
     
Given the context excerpt determine participation type of the character **{character}** whose mention is highlighted in bold in the target phrase..
In both the target phrase and the context, the mention of the character is marked in bold. 
The mention does not necessarily have to be the character’s full name — it may also appear as a pronoun or any other referring expression.
If no text is highlighted in bold in either the phrase or the context, this indicates an outside mention: the character is referenced in the broader context but not explicitly mentioned within the target phrase itself. In this case, the relevant reference lies outside the phrase, and therefore no bold marking appears inside it.

Carefully follow the instructions and select strictly one label.

""".strip()
             }  
                 
             
    ]
        

    types_description = """
Important:
Do not decide labels based on grammatical voice (active/passive).
To assign the correct label (agentive, low_agentive, or passive), 
follow this step-by-step logic:

STEP 1:
Ask whether the action is **physically** performed or **physically** carried out by the figure associated with the annotation, or whether the capability for this action is attributed to them. 
Important: grammatical passive (e.g., “was discovered”) does NOT automatically mean the label "passive".
– If the figure does not perform the action and is only affected by it (or if the action by/for the figure is negated), assign the label: passive.
– If the figure performs the action (even if the agent is implicit), proceed to STEP 2.

STEP 2: 
If the action is not actually performed but only imaganed, wished or hypothetical, asign the lable low_agentive. Otherwise, proceed to STEP 3.

STEP 3:
Ask whether the action is consciously or intentionally caused, controlled, or influenced by the figure. 
Heuristic: if the text suggests that the character could stop or interrupt the activity, it counts as controlled.
– If yes, assign the label: agentive.
– If no or if control is not explicitly indicated, assign the label: low_agentive.
""" 
    

    all_phrase_labels = {
        'agentive': {
            'gold': [],
            'predicted': []
        },
        'low_agentive': {
            'gold': [],
            'predicted': []
        },
        'passive': {
            'gold': [],
            'predicted': []
        },
    }
    

    schema = AgentiveResponse.model_json_schema() # getting python-dict of the response-schema

    sampling_params = SamplingParams( # preparing config-object with the params
        temperature = 0,
        max_tokens = 1024,
        structured_outputs = StructuredOutputsParams(json = schema) 
    )

    
    mlb = MultiLabelBinarizer(classes = instance["all_text_chars"])
    mlb.fit([[]])
    
    for category in category_list:
        for char in instance["golden_labels_with_chars"][category]:
            all_phrase_labels[category]['gold'].append(char)
        
    
    phrase_spans = instance["phrase_spans"]
    context_spans = instance["context_spans"]
    
    for char in instance["predicted_char_list"]:
        #marking the character mention in the PHRASE with asterisks:
        char_mention_spans = get_mention_spans(phrase_spans, doc_annotations, author, char) #get char mention spans overlapping the phrase 
        processed_phrase = []                   # будет СПИСОК кусков уже полностью готовой фразы, который я объединю сепаратором "..."
        for p_span in phrase_spans:             # последовательно обрабатываем каждый кусок phrase
            marking_spans = []                  # спаны, которые надо выделить звёздами в текущем куске фразы
            p_start, p_end = p_span             # спан куска фразы
            for m_span in char_mention_spans:   # сравниваем кусок фразы со всеми кусками мэншенов
                overlap_span = intersect_spans(p_span, m_span)
                if (overlap_span):
                    overlap_span = [v - p_start for v in overlap_span] #нормализуем отрезок: полагаем, что ноль - это начало куска фразы
                    marking_spans.append(overlap_span) #уже нормализованные спаны которые надо заключить в **
            #наконец передаём кусок фразы (в текстовом виде) и пронормированные спаны для звёзд в функцию:
            part = mark_text_with_asterisks(doc_annotations["text"][p_start:p_end], marking_spans)
            processed_phrase.append(part)
        processed_phrase = ' ... '.join(processed_phrase)
        processed_phrase = processed_phrase.replace('\r', '').replace('\n', ' ') 
        
        
        #marking the character mention in the CONTEXT with asterisks:
        processed_context = []                   # будет СПИСОК кусков уже полностью готовой фразы, который я объединю сепаратором "..."
        for sentence_span in context_spans:             # последовательно обрабатываем каждое предложение контекста
            marking_spans = []                  # спаны, которые надо выделить звёздами в текущем куске контекста
            s_start, s_end = sentence_span             # спан куска контекста (т.е. спан предложения)
            for m_span in char_mention_spans:   # сравниваем кусок фразы со всеми кусками мэншенов
                overlap_span = intersect_spans(sentence_span, m_span)
                if (overlap_span):
                    overlap_span = [v - s_start for v in overlap_span] #нормализуем отрезок: полагаем, что ноль - это начало куска фразы
                    marking_spans.append(overlap_span) #уже нормализованные спаны которые надо заключить в **
            #наконец передаём кусок контекста (в текстовом виде) и пронормированные спаны для звёзд в функцию:
            part = mark_text_with_asterisks(doc_annotations["text"][s_start:s_end], marking_spans)
            processed_context.append(part)
        processed_context = ' '.join(processed_context)
        processed_context = processed_context.replace('\r', '').replace('\n', ' ')     
            
    
        #filling template for the current prompt with the real data:
        complete_msg = [] # make it empty
        complete_msg.append(template_messages[0])
        complete_msg.append(
            {
                "role": "user",
                "content":
                    template_messages[1]["content"].format( # для обычного промпта без прикреплённого пдф
                    character = char,
                    context_text = processed_context,
                    phrase_text = processed_phrase,
                    labels_description = types_description
                    )
            } 
        )
            
            
        attempt = 0
        
        while attempt < MAX_RETRIES:
            try:     
                #sending the prompt to the model:  
                # response = client.beta.chat.completions.parse(
                #     #model="gpt-5-mini", # TODO: check the name of the model
                #     model = "gpt-5.2",
                #     messages = complete_msg,
                #     response_format = AgentiveResponse
                # )
                
                response = llm.chat(
                    [complete_msg],
                    sampling_params
                )
                break
            
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    attempt += 1
                    continue
                else: 
                    print(f"[X] Unexpected error: {e}")
                    break
                    
        #converting the response to the json format:
        #cleaned_response = response.choices[0].message.parsed.model_dump()
        text_response = response[0].outputs[0].text # getting text of the response
        object_response = AgentiveResponse.model_validate_json(text_response) # validating and getting schema-object of json (AgentiveResponse object)
        cleaned_response = object_response.model_dump() # getting python-dict
        
        if cleaned_response == None:
            print(f'[X] Cleaned response is NONE')  
        #TODO: сделать проверку на соответствие лейбла одному из допустимых значений.
        else:
            label = cleaned_response["prediction"]
            all_phrase_labels[label]["predicted"].append(char)
#------------------------------- цикл кончился---------------------------------   
    
    binarized_all_phrase_labels = {
        category: {
            k: mlb.transform([v])[0].tolist()
            for k, v in inner_dict.items()
        }
        for category, inner_dict in all_phrase_labels.items()
    }
        
    #----------------------Логированипе в джейсон файлы------------------------
    log_entry = {
        'idx': instance['idx'],
        'doc_title': instance['doc_title'],
        'phrase_spans': instance['phrase_spans'],
        'phrase_text': instance["phrase_text"],
        #'context_text': processed_context,
        'labels_data': all_phrase_labels,
        'binarized_labels_data': binarized_all_phrase_labels
    }
    
    json.dump(log_entry, pretty_log_file, ensure_ascii = False, indent = 4) # Логированипе в красивый json файл для ручного просмотра:
    
    json.dump(log_entry, machine_log_file, ensure_ascii = False)  # Логированипе в jsonlines файл для дальнейшего автоматического разбора 
    machine_log_file.write("\n") # обязательно добавлять \n чтобы соблюсти правила jsonlines!
    print(f'[TEXT]: {instance["doc_title"]}, [ID]: {instance["idx"]}')
    
    
    # #------------------Логирование в текстовый файл более подробно:------------
    # text_log_file.write("=" * 80 + "\n")
    # text_log_file.write(f"[ID]     {instance['idx']}\n")
    # #text_log_file.write(f"[PHRASE SPANS]        {instance['phrase_spans']}\n")
    # text_log_file.write(f"[PHRASE TEXT]        {processed_phrase}\n")
    # #text_log_file.write(f"[CONTEXT SPANS]        {instance['context_spans']}\n")
    # text_log_file.write(f"[CONTEXT TEXT]        {processed_context}\n")
    # text_log_file.write(f"[GOLD LABEL]    {golden_label}\n")
    # text_log_file.write(f"[PREDICTION]    {cleaned_response['prediction']}\n")
    # text_log_file.write(f"[EXPLANATION]    {cleaned_response['explanation']}\n")
    # text_log_file.write("\n")
    
    # text_log_file.write("[PROMPT]\n")
    # text_log_file.write("-" * 80 + "\n")
    # text_log_file.write(complete_msg[1]['content'] + "\n")
    # text_log_file.write("-" * 80 + "\n\n")
    
    # #text_log_file.write("[EXPLANATION]\n")
    # #text_log_file.write(cleaned_response["explanation"] + "\n")
    # text_log_file.write("=" * 80 + "\n\n")
    
    return True


        

    



#--is not being used anymore
def parse_gpt_response(response):
    import json

    try:
        content_str = response.choices[0].message.content
        content_json = json.loads(content_str)

        explanations = [step["explanation"] for step in content_json.get("steps", [])]
        final_answer = content_json.get("final_answer", "")

        return {
            "explanations": explanations,
            "final_answer": final_answer
        }

    except Exception as e:
        print("Error while proccesing the answer:", e)
        return None
    


#TODO: edit this function due to the new formulation (generalization of the task)
def log_statistics(log_file_path):    
    label_map = {
    "passive": 0,
    "agentive": 1,
    "low_agentive": 2
    }
    reverse_label_map = {v: k for k, v in label_map.items()}
    with open(log_file_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
        y_true = []
        y_pred = []
        for instance in logs:
            y_true.append(label_map[instance["golden_label"]])
            y_pred.append(label_map[instance["prediction"]])
    target_names = ['passive', 'agentive', 'low_agentive']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    true_named = Counter({reverse_label_map[k]: v for k, v in Counter(y_true).items()})
    pred_named = Counter({reverse_label_map[k]: v for k, v in Counter(y_pred).items()})
    # print("True label distribution:", Counter(y_true))
    # print("Predicted label distribution:", Counter(y_pred))
    # print("True label distribution:", true_named)
    # print("Predicted label distribution:", pred_named)
    ordered_labels = ['passive', 'agentive', 'low_agentive']
    
    print("True label distribution:")
    for label in ordered_labels:
        print(f"  {label}: {true_named[label]}")
    
    print("Predicted label distribution:")
    for label in ordered_labels:
        print(f"  {label}: {pred_named[label]}")
    
    
    y_true_counter = Counter([entry["golden_label"] for entry in logs])
    y_pred_counter = Counter([entry["prediction"] for entry in logs])
    # Упорядочим по меткам
    labels = ["passive", "low_agentive", "agentive"]
    y_true_total = [y_true_counter[label] for label in labels]
    y_pred_total = [y_pred_counter[label] for label in labels]
    
    # Построим графики
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y_true_total, width, label="Golden", color="skyblue")
    ax.bar([i + width for i in x], y_pred_total, width, label="Predicted", color="orange")
    ax.set_ylabel("Count")
    ax.set_title("Label Distributions: Golden vs Predicted")
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels)  # это задаёт подписи по оси X
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    
    #+++++++scikit-learn metrics:++++++++
    print("\n+++++++scikit-learn metrics:+++++++\n")
    labels = [0, 1, 2]  # явный порядок меток (рекомендуется)
    # 1) Accuracy (percentage of the correct ones)
    acc = accuracy_score(y_true, y_pred)    
    print(f'Accuracy: {acc}')
    
    # 2) Precision / Recall / F1 with MACRO:
    prec_macro  = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    rec_macro   = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    print(f'Precision: {prec_macro}')
    print(f'Recall: {rec_macro}')
    print(f'F1: {f1_macro}')



# постпроцессим список чаров, которые вернула модель: убираем пробелы, оставляем только строки, убираем дубли..
def normalize_char_list(char_list):
    result_set = set()
    for char in char_list:
        if isinstance(char, str):   # берём только строки
            name = char.strip().lower()     # убираем пробелы по бокам и к нижнему регистру приводим
            if name:                    # только непустые рассматриваем!
                result_set.add(name) # добавляем в наш итоговый сэт
    
    return result_set



# mode = 'participants' - for the 1st group of prompts (regarding chars who participate in the phrase)
# mode = 'labels' - for the 2d group of prompts (regarding particular label of a character)
def calc_metrics(log_file_name, mode):
    
    FP = FN = TP = 0
    logs = []
    with open(log_file_name, 'r', encoding = 'utf-8') as f: # загружаем данные из файла
        for line in f:
            obj = json.loads(line)
            logs.append(obj)
    
            
    if mode == 'participants':
        for obj in logs:
            gold_char_set = normalize_char_list(obj["gold_char_list"])
            pred_char_set = normalize_char_list(obj["predicted_char_list"])
            TP += len(gold_char_set & pred_char_set)
            FP += len(pred_char_set - gold_char_set)
            FN += len(gold_char_set - pred_char_set)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
        print("--------------OVERALL METRICS FOR CHAR LISTS:--------------")
        print(f"  Precision: {precision}")
        print(f'  Recall: {recall}')
        print(f'  F1: {f1} \n\n')
    
    elif mode == 'labels':
        y_true = []
        y_pred = []
        for obj in logs:
            golden_label = obj["golden_label"] # может несколько лейблов у перса быть в рамках одной фразы, хотя и маловероятно
            predicted_label = obj["predicted_label"]
            y_true.append(label_map[golden_label])
            y_pred.append(label_map[predicted_label])
        target_names = ['passive', 'agentive', 'low_agentive']
        print("--------------OVERALL METRICS FOR LABELS:--------------\n")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        true_named = Counter({reverse_label_map[k]: v for k, v in Counter(y_true).items()})
        pred_named = Counter({reverse_label_map[k]: v for k, v in Counter(y_pred).items()})
        ordered_labels = ['passive', 'agentive', 'low_agentive']

        print("True label distribution:")
        for label in ordered_labels:
            print(f"  {label}: {true_named[label]}")
        
        print("Predicted label distribution:")
        for label in ordered_labels:
            print(f"  {label}: {pred_named[label]}")
            
        y_true_counter = Counter([entry["golden_label"] for entry in logs])
        y_pred_counter = Counter([entry["predicted_label"] for entry in logs])
        # Упорядочим по меткам
        labels = ["passive", "low_agentive", "agentive"]
        y_true_total = [y_true_counter[label] for label in labels]
        y_pred_total = [y_pred_counter[label] for label in labels]
        
        # Построим графики
        x = range(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, y_true_total, width, label="Golden", color="skyblue")
        ax.bar([i + width for i in x], y_pred_total, width, label="Predicted", color="orange")
        ax.set_ylabel("Count")
        ax.set_title("Label Distributions: Golden vs Predicted")
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(labels)  # это задаёт подписи по оси X
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        #+++++++scikit-learn metrics:++++++++
        print("\n+++++++scikit-learn metrics:+++++++\n")
        labels = [0, 1, 2]  # явный порядок меток (рекомендуется)
        # 1) Accuracy (percentage of the correct ones)
        acc = accuracy_score(y_true, y_pred)    
        print(f'Accuracy: {acc}')
        
        
        # Precision / Recall / F1 with MACRO:
        prec_macro  = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        rec_macro   = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        f1_macro    = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        
        print(f'Precision: {prec_macro}')
        print(f'Recall: {rec_macro}')
        print(f'F1: {f1_macro}')
        
        
    else: 
        raise ValueError(f'Incorrect mode value for calc_metrics: {mode}')
    



def init_logs():
    
    # clean all the log files:
    for log_folder_path in [CHAR_LISTS_LOG_DIR, LABELS_LOG_DIR]:    
        for file_name in os.listdir(log_folder_path):
            file_path = os.path.join(log_folder_path, file_name)
            if os.path.isfile(file_path):
                open(file_path, "w", encoding="utf-8").close()
    
    # clean data for prompts log file:
    prompt_data_log_path = os.path.join(BASE_LOG_DIR, aux_prompt_data_file)
    open(prompt_data_log_path, "w", encoding="utf-8").close()
    
    
    
def do_char_list_prompting(prompt_data): # в параметры передать контекст_энд_чарс

    succesful_prompts = 0 # succesful prompts
    failed_prompts = 0 # failed prompts 
    all_log_entries = []

    # for logs (3 log files):
    char_lists_pretty_path = os.path.join(CHAR_LISTS_LOG_DIR, pretty_log_name) 
    char_lists_machine_path = os.path.join(CHAR_LISTS_LOG_DIR, machine_log_name)
    char_lists_full_path = os.path.join(CHAR_LISTS_LOG_DIR, full_log_name)
    
    with (
            open(char_lists_pretty_path, "a", encoding="utf-8") as t1_pretty,
            open(char_lists_machine_path, "a", encoding="utf-8") as t1_machine,
            open(char_lists_full_path, "a", encoding="utf-8") as t1_text,
    ) :
        for instance in prompt_data:
            is_ok, log_entry = Send_Char_List_Prompt(instance, t1_pretty, t1_machine, t1_text)
            if is_ok:
                succesful_prompts += 1  
                all_log_entries.append(log_entry)
            else: 
                print(f'Cleaned response is NONE for the phrase: **{instance["phrase_text"]}** ')
                failed_prompts += 1
            
    print(f'   [CHAR LIST] Total succesful prompts: {succesful_prompts}')
    print(f'   [CHAR LIST] Total failed prompts: {failed_prompts} \n\n')
    
    return all_log_entries
    
    
    

def do_label_prompting(extended_prompt_data, doc_store):
    
    succesful_prompts = 0 # succesful prompts
    failed_prompts = 0 # failed prompts 
    
    labels_pretty_path = os.path.join(LABELS_LOG_DIR,pretty_log_name)
    labels_machine_path = os.path.join(LABELS_LOG_DIR, machine_log_name)
    labels_full_path = os.path.join(LABELS_LOG_DIR, full_log_name)
    
    with (
            open(labels_pretty_path, "a", encoding="utf-8") as t2_pretty,
            open(labels_machine_path, "a", encoding="utf-8") as t2_machine,
            open(labels_full_path, "a", encoding="utf-8") as t2_text,
    ) :
        for instance in extended_prompt_data:
            doc_title = instance["doc_title"]
            doc_annotations = doc_store[doc_title]
            is_ok = Send_Labels_Prompt(instance, doc_annotations, t2_pretty, t2_machine, t2_text)
            if is_ok:
                succesful_prompts += 1               
            else: 
                print(f'Cleaned response is NONE for the phrase: **{instance["phrase_text"]}** ')
                failed_prompts += 1

    print(f'   [LABELS] Total succesful prompts: {succesful_prompts}')
    print(f'   [LABELS] Total failed prompts: {failed_prompts} \n\n')

    



def main():
    
    json_files = [f for f in os.listdir(folder_data_path) if f.endswith('.json')]
    doc_store = {}
    for file in json_files:
        with open(f'data/{file}', "r", encoding="utf-8") as f:
            data = json.load(f)
        full_annotations = combining_annotations_by_authors(data)
        title = full_annotations['title']
        doc_store[title] = full_annotations
        
    
    
    # num_file = 1
    # total_cur_errors = 0 
    # total_cur_char_list_prompts = 0 
    # total_cur_labels_prompts = 0
    # total_failed_prompts = 0
    # total_data_instances = 0 # total number of data objects for prompts, not equal to total prompts
    
    init_logs() # зачищаем старые файлы, которые через режим "а" дописываются у нас 
    
    prompt_data = prepare_prompt_data()  # getting prepared info for prompts for N participation objects
    char_list_answers = do_char_list_prompting(prompt_data)
    
    
    
    log_path_participants = os.path.join(CHAR_LISTS_LOG_DIR, machine_log_name)
    log_path_labels = os.path.join(LABELS_LOG_DIR, machine_log_name)
    
    # char_list_answers = []
    # with open(log_path_participants, 'r', encoding='UTF-8') as f:
    #     for line in f:
    #         obj = json.loads(line)
    #         log_entry = {
    #             'idx': obj['idx'],
    #             'doc_title': obj['doc_title'],
    #             'phrase_spans': obj['phrase_spans'],
    #             'phrase_text': obj['phrase_text'],
    #             'context_spans': obj['context_spans'],
    #             'context_text': obj['context_text'],
    #             'all_text_chars': obj['all_text_chars'],
    #             'gold_char_list': obj['gold_char_list'],
    #             "golden_labels_with_chars": obj["golden_labels_with_chars"],
    #             'predicted_char_list': obj["predicted_char_list"],    
    #         }
    #         char_list_answers.append(log_entry)
    
    
    
    do_label_prompting(char_list_answers, doc_store)
          

    # print("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++:")
    # print("  TOTAL:")
    # print(f'  [CHAR LISTS] Total prompts: {total_cur_char_list_prompts}')
    # print(f'  [LABELS] Total prompts: {total_cur_labels_prompts}')
    # print(f'  [ALL] Total failed prompts {total_failed_prompts} \n\n')
    

    
    #calc_metrics(log_path_participants, 'participants')
    #calc_metrics(log_path_labels, 'labels')
   




    
main()

# data = json.load(open("Full_file.json"))
# text = data["text"]
# start = int(input("Enter start of span: "))
# end = int(input("Enter end of span: "))
# print(text[start:end])