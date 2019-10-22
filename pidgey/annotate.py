import random
import functools
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, Dropdown, HTML, HBox, IntSlider, FloatSlider, Textarea, Output
from sklearn.model_selection import train_test_split



class Annotator:
    def __init__(self, examples, options=None, shuffle=False, include_skip=True, max_options_dropdown=10, 
                        show_columns=None, annotation_column_name='annotation', display_fn=display):

        self.options = options
        self.shuffle = shuffle
        self.include_skip = include_skip
        self.max_options_dropdown = max_options_dropdown
        self.show_columns = show_columns
        self.annotation_column_name = annotation_column_name
        self.display_fn = display
        self.current_index = -1
        self.current_train_score = 0.0
        self.current_test_score = 0.0

        examples = examples.copy()

        if not isinstance(examples, pd.DataFrame):
            self.examples = pd.DataFrame({'examples': examples})
            self.show_columns = ['examples']

        if shuffle:
            examples = examples.sample(frac=1.0).reset_index(drop=True)

        if annotation_column_name not in examples.columns:
            examples[annotation_column_name] = None
        self.examples = examples     

        self.pipeline = None


        if type(self.options) == list:
            task_type = 'classification'
        elif type(self.options) == tuple and len(options) in [2, 3]:
            task_type = 'regression'
        elif self.options is None:
            task_type = 'captioning'
        else:
            raise Exception('Invalid options')
        self.task_type = task_type
        
    def add_pipeline(self, pipeline, fit_every_n_sample=10, fit_minimum_samples=50, fit_test_size=0.2, fit_features='all'):
        self.pipeline = pipeline
        self.fit_every_n_sample = fit_every_n_sample
        self.fit_minimum_samples = fit_minimum_samples
        self.fit_test_size = fit_test_size
        self.fit_features = [f for f in fit_features if f in self.examples.columns]

    def init_html(self):
        self.count_label = HTML()


    def set_label_text(self):
            self.count_label.value = '{} examples annotated, {} examples left <ul><li>current train score {}</li><li>current test score  {}</li></ul>'.format(
                len(self.examples) - self.examples[self.annotation_column_name].isnull().sum(), 
                len(self.examples) - self.current_index,
                self.current_train_score,
                self.current_test_score
            )

    def show_next(self):
        self.current_index += 1
        self.set_label_text()
        if self.current_index >= len(self.examples):
            for btn in self.buttons:
                btn.disabled = True
            print('Annotation done.')
            return

            #current index should be len(examples.dropna(subset=['label']))
        if self.pipeline is not None \
            and self.current_index >= self.fit_minimum_samples \
            and (self.current_index - self.fit_minimum_samples) % self.fit_every_n_sample == 0:
                self.fit_pipeline()
        with self.out:
            clear_output(wait=True)
            self.display_fn(self.examples.loc[[self.current_index]][self.show_columns])

    def fit_pipeline(self):
            samples = self.examples.copy().dropna(subset=[self.annotation_column_name])
            if self.fit_features == 'all':
                X = samples.drop(self.annotation_column_name, axis=1)
            else:
                X = samples[self.fit_features]
            y = samples[[self.annotation_column_name]].values.ravel()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.fit_test_size)        
            self.pipeline.fit(X_train, y_train)
            self.current_train_score = self.pipeline.score(X_train, y_train)
            self.current_test_score  = self.pipeline.score(X_test, y_test)
            self.set_label_text()

    def add_annotation(self, annotation):
        self.examples.at[self.current_index, self.annotation_column_name] = annotation
        self.show_next()

    def skip(self, btn):
        self.show_next()
    

    def annotate(self):
        self.count_label = HTML()
        self.set_label_text()
        display(self.count_label)
        self.buttons = []
        
        if self.task_type == 'classification':
            use_dropdown = len(self.options) > self.max_options_dropdown

            if use_dropdown:
                dd = Dropdown(options=self.options)
                display(dd)
                btn = Button(description='submit')
                def on_click(btn):
                    self.add_annotation(dd.value)
                btn.on_click(on_click)
                self.buttons.append(btn)
            
            else:
                for label in self.options:
                    btn = Button(description=label)
                    def on_click(label, btn):
                        self.add_annotation(label)
                    btn.on_click(functools.partial(on_click, label))
                    self.buttons.append(btn)

        elif self.task_type == 'regression':
            target_type = type(options[0])
            if target_type == int:
                cls = IntSlider
            else:
                cls = FloatSlider
            if len(options) == 2:
                min_val, max_val = options
                slider = cls(min=min_val, max=max_val)
            else:
                min_val, max_val, step_val = options
                slider = cls(min=min_val, max=max_val, step=step_val)
            display(slider)
            btn = Button(description='submit')
            def on_click(btn):
                add_annotation(slider.value)
            btn.on_click(on_click)
            buttons.append(btn)

        else:
            ta = Textarea()
            display(ta)
            btn = Button(description='submit')
            def on_click(btn):
                self.add_annotation(ta.value)
            btn.on_click(on_click)
            self.buttons.append(btn)

        if self.include_skip:
            btn = Button(description='skip', button_style='info')
            btn.on_click(self.skip)
            self.buttons.append(btn)

        self.box = HBox(self.buttons)
        display(self.box)

        self.out = Output()
        display(self.out)

        self.show_next()

    def get_annotated_examples(self):
        return self.examples.dropna(subset=[self.annotation_column_name])