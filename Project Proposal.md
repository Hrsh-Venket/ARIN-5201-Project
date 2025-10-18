# Project Proposal: Creative Poster Generator

## Group Member information
VENKET, Hrsh (21247584, MAI)
SHAO, Zixuan (21254032, > MAIE)
LI, Ang (21273650, MAIE)
TOLEGENULY, Ibragim (21247314, MAI)

## Project Background
For this project, we will be orchestrating agents to do Poster Generator. Below are the input and output of our pipeline. We will use LangGraph to orchestrate a set of agents that will help construct the creative poster. Using different specialised agents for the different tasks can give better performance. Further, while our project will certainly meet the requirements given to us, we want to build a tool that can go beyond it. For a real user of a tool like this, AI may not be the best way to get their exact vision for a poster. Further, Generative AI models (despite many checks) are prone to errors and hallucinations. Thus, a far more useful tool would not only create a poster, but also give the user the constituent assets so they can adjust the poster according to what they want

### Input: 

- 2-5 keywords (required)
- Company Logo (optional)
- Other Marketing Material (optional)

### Output:

- 720 x 1280 poster png image

(we will try to implement these if we are abl eto build the original pipeline without issues)

- Elements used in poster in a zip file
- layout.json — list of elements with {type, x, y, w, h, z, constraints,textStyle}
- style.json — palette (primary/secondary, neutrals), type roles (display/headline/body/label), spacing scale
- Optional layout.svg (wireframe) for quick preview

## Preliminary Methodology

Below are the list of agents we intend to use and some description and ideas regarding them. We may adjust this plan as we are building the project

1. Planning and Prompting Agent:
  - Interpret the keywords and create a plan for the poster content
  - Work with the other agents (particularly brand grounding agent, and controllable layout generator to create a suitable plan)
  - Prompt the asset generation agents

2. Brand Grounding Agent:
  - Ensure coherent branding decisions
  - Extract pallete and font from optional logo and marketting Material
  - Ensure that the assets are generated based on the input keywords

3. Controllable layout Generator
  - We would like to use the publicly available code from [LayoutDM: Discrete Diffusion Model for Controllable Layout Generation (CVPR2023)](https://cyberagentailab.github.io/layout-dm/) along with model weights or [Bidirectional Layout Encoder (ECCV'22)](https://github.com/google-research/google-research/tree/master/layout-blt) which has the same.

4. Image Asset Generating agent
  - We intend to use do image generation along with the alpha channel. We have a [publicly available model on Hugging Face](https://huggingface.co/rootonchair/diffuser_layerdiffuse) of this, but will explore other open source and closed source alternatives.
  - Using the alpha channel when generating image assets makes putting elements together easier

5. Text Asset Generating Agent
- This can be done by the same model as the planning agent, but it is worth distinguishing as this task may need to be repeated (make text longer or shorter as a design decision, change content so it is more connected to the prompt)

6. Editing Agent:
- Put together the generated assets according to decisions made by planning agent
- Trigger previous agents to re-generate a subset of the assets
- Ensure that the assets are made according to the specifications made by the planning agent
- Ensure that a coherent poster is made and checks by the evaluation agent are passed
- Here, we intend to use an MCP protocol to give the LLM access to basic editing tools (so that it can make it's own aesthetic decisions beyond the instructions by the planning agent). We have found the editing tool GIMP has a [very well documented API](https://developer.gimp.org/api/3.0/), with which we can build an MCP for the editing agent

7. Evaluation agent:
- Create some automatic checks to see if poster needs to be redesigned. We have listed a few examples below:
  - No text in the margins 
  - Heuristics to check readability (including contrast, number of colours in poster, etc)
  - Font consistency
  - Use this aesthetic score predictor to check the appearance of the 

## Project Timeline



