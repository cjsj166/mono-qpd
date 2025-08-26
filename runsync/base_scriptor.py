import datetime
from pathlib import Path
from jinja2 import Template
import os
from dataclasses import asdict

class BaseScriptor:
    def __init__(self):
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.default = {}
        self.custom_setting = {}

        self.script_template = ''''''
        self.script = ''
        self.exec_path = Path(__file__).parent.parent.resolve()

    def make_templates(self):
        self.make_script_template()
        return self.script_template

    def render_template(self):
        self.set_context()
        script = Template(self.script_template).render(context=self.context)
        self.script = script
        return script
    
    def write_script(self): # You can override in subclass
        self.job_script_path = Path(self.exec_path) / "scripts" / f"{self.timestamp}.sh"

        os.makedirs(self.job_script_path.parent, exist_ok=True)
        with open(self.job_script_path, "w") as f:
            f.write(self.script)
        return self.job_script_path
    
    def get_args(self):
        raise NotImplementedError("Subclasses must implement get_args method")

    def set_context(self):
        raise NotImplementedError("Subclasses must implement set_context method")

    def make_script_template(self):
        raise NotImplementedError("Subclasses must implement _make_header method")

