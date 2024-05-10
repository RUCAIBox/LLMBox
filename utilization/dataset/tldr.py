from ..metric import Rouge
from .generation_dataset import GenerationDataset


class TLDR(GenerationDataset):
    """The dataset of tl;dr

    The TL;DR Dataset is an English-language dataset containing
    Reddit posts to summarize.

    Examples:
        prompt: SUBREDDIT: r/loseit TITLE: SV & NSV! Keeping on keeping on. POST: 30F, 5'6". SW: 236 GW: 150 CW: 219 I weigh myself weekly and measure myself monthly. I'd hit a plateau the last four weeks or so where I was stuck at 222. Felt like kind of a bummer, but knew it's because I haven't been as strict as I should with my diet, and the last week and a half have been crazy with life things, so I haven't been exercising as frequently as I've gotten used to. When I weighed myself as normal on Monday, I was kind of disappointed to see the scale not budging and figured it was time to buckle down again and really watch my diet. Today was my measure-in day, and I've felt cruddy in general since Monday because I caught some chest congestion/cold bug over the weekend. I get on the scale...it says 219. Whaaaaat? I take my measurements, which are down slightly from last month, and with an total-body loss of 8 inches from my starting point on 12/23/14! Some of my clothes have been feeling a bit looser as of late and now I know it's just not in my head. I'm now the lightest and smallest I've been since right around high school! TL;DR:
        label: Progress is still happening, even when you think it might not be! Don't get discouraged, even if your journey seems to be going slowly. Don't give up, warriors.
    """

    instruction = "{source}"
    evaluation_set = "train"
    example_set = None
    metrics = [Rouge()]
    load_args = ("CarperAI/openai_summarize_tldr",)
    extra_model_args = dict(temperature=0)

    def format_instance(self, instance):
        source = instance["prompt"]
        target = instance["label"]
        return dict(source=source, target=target)

    @property
    def references(self):
        return [instance["label"][:] for instance in self.evaluation_data]
