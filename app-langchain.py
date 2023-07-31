import gradio as gr
import sqlite3
import re
import asyncio
import yaml
from utils.callbacks import FinalOutputAsyncHandler
from utils.sql_agent import get_sql_agent_executor
from utils.open_api import get_openai_llm
from anyio.from_thread import start_blocking_portal
import pandas as pd
from helper import get_table_html, create_db, sanitize_tablename

config = yaml.safe_load(open("config.yaml"))
llm = get_openai_llm()
example_dataframe = pd.read_csv("example.csv")


def on_download(chat_history, file_component):
    with open(config["HISTORY_FILENAME"], 'w') as f:
        f.write(str(chat_history))
    return gr.update(value=config['HISTORY_FILENAME'])


async def language_modeling_bot(history, agent_ex):
    history[-1][1] = ''
    user_message = history[-1][0]
    queue = asyncio.Queue()
    event = asyncio.Event()

    async def task(prompt):
        result = await agent_ex.acall(prompt, callbacks=[FinalOutputAsyncHandler(queue, event)])
        return result
    with start_blocking_portal() as portal:
        portal.start_task_soon(task, user_message)
        while not queue.empty() or not event.is_set():
            next_token = await queue.get()
            history[-1][1] += next_token
            yield history

CSS = ''

with open('./bootstrap.min.css', 'r') as f:
    CSS += f.read()

with open('style.css', 'r') as f:
    CSS += f.read()

title = "Language Modeling Bot"

demo = gr.Blocks(title=title, css=CSS)

with demo:

    # State
    agent_executor = gr.State(value=None)
    db = gr.State(value=None)

    # Frontend
    gr.HTML('''<img class = "d-block p-3" src = '', alt = "logo">''',
            elem_id=['logo'])

    with gr.Tab('Chatbot') as main_tab:
        gr.HTML('''<div class = "title-sql" Querying tabular data</div>''')
        gr.HTML('''<ol>
                <li>Upload a CSV file</li>
                <li> Ask a question</li>
                <li> Save chat history</li>
                </ol>
        ''')
        example_btns = []
        with gr.Row(elem_id='example-view') as example_view:
            for x in config['EXAMPLES']:
                example_btns.append(
                    gr.Button(x, show_label=False, elem_id='example-btn'))

        with gr.Column(elem_id='chatbot-group') as chatbot_group:
            chatbot = gr.Chatbot(visible=False)
            user_input = gr.Textbox(
                show_label=False, placeholder='Ask a question', elem_id='user-input')

        with gr.Row(visible=False) as history_group:
            clear = gr.Button('Clear History', elem_classes=['custom-btn'])
            download_chat = gr.Button(interactive=False, visible=False)
        downloadable = gr.File(interactive=False, visible=False)

    with gr.Tab('Add Data'):
        with gr.Row():
            upload_button = gr.UploadButton(
                "Add Files to Knowledge Base",
                file_types=["text"],
                file_count="single",
                elem_classes=['custom-btn'],
            )
        with gr.Row():
            usable_tables = gr.Dropdown(
                label = 'View Sample Data',
                interactive = True
            )
        data_view = gr.HTML(label = 'Data View', elem_id = 'example-view')
    
    # Event Handlers

    def init(database, agent_ex):
        database = create_db(tablename='example', dataframe=example_dataframe)
        agent_ex = get_sql_agent_executor(llm=llm, db=database, top_k=10, verbose=True)
        return database, agent_ex, gr.Dropdown.update(choices = list(database.get_usable_table_names()))
    
    demo.load(
        init, [db, agent_executor], [db, agent_executor, usable_tables]
    )

    def update_agent(agent_ex, database):
        agent_ex = get_sql_agent_executor(llm=llm, db=database, top_k=10, verbose=True)
        return {agent_executor: agent_ex, db: database}
    
    def language_modeling_user(user_message, history):
        if len(user_message) == 0:
            raise gr.Error("Please enter a question")
        return {user_input: '', chatbot: history + [[user_message, None]], history_group: gr.update(visible=True)}

    def on_upload(file, database, agent_exec, progress = gr.Progress()):
        progress(0, desc = "Uploading file...")
        filename = os.path.split(file.name)[-1]
        if not filename.endswith('.csv'):
            raise gr.Error("Only CSV files are supported")
        df = pd.read_csv(file)
        progress(50, desc = "Creating database...")
        tablename = sanitize_tablename(filename)
        database = create_db(tablename=tablename, dataframe=df)
        agent_exec = get_sql_agent_executor(llm=llm, db=database, top_k=10, verbose=True)
        progress(100, desc = "Done!")
        return {db: database, agent_executor: agent_exec, usable_tables: gr.Dropdown.update(choices = list(database.get_usable_table_names()))}

    # Event Listeners

    for btn in example_btns:
        btn.click(
            lambda: {chatbot: gr.update(visible=True), example_view: gr.Row.update(visible=False)},
            None,
            [chatbot, example_view]
        ).success(
            language_modeling_user,
            [btn, chatbot],
            [user_input, chatbot, history_group],
            queue = False
        ).success(
           language_modeling_bot,
           [chatbot, agent_executor],
           [chatbot]
        )
    
    upload_button.upload(
        lambda: [gr.update(value = config['EXAMPLES_GENERIC'][i]) for i in range(len(example_btns))],
        None,
        [btn for btn in example_btns]
    ).success(
        lambda: {chatbot: gr.update(visible=False), example_view: gr.update(visible=True)},
        None,
        [chatbot, example_view]
    ).success(
        on_upload,
        [upload_button, db, agent_executor],
        [db, agent_executor, data_view, usable_tables]
    )

    user_input.submit(
        lambda: {chatbot: gr.update(visible=True)},
        None,
        chatbot
    ).success(
        language_modeling_user,
        [user_input, chatbot],
        [user_input, chatbot, history_group],
        queue = False
    ).success(
        language_modeling_bot,
        [chatbot, agent_executor],
        [chatbot]
    )

    clear.click(
        lambda: [None, gr.update(visible=False), gr.update(visible=False)],
        None,
        [chatbot, history_group, downloadable]
    ).success(
        lambda: {chatbot: gr.update(visible=False)},
        None,
        chatbot
    )

    download_chat.click(
        on_download,
        [chatbot, downloadable],
        [downloadable]
    ).success(
        lambda: {downloadable: gr.update(visible=True)},
        None,
        downloadable
    )

    usable_tables.select(
        get_table_html,
        usable_tables,
        data_view
    )


demo.queue(
    concurrency_count = 4,
    max_size = 4
).launch(
    server_name = "0.0.0.0",
    server_port = 7878,
    share = True
)
