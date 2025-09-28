from .vocab import load_vocab, DEFAULT_PATH
import json

VOCAB = load_vocab(DEFAULT_PATH)
vocab_json = json.dumps(VOCAB)

SCHEMA_JSON = """
{
  "meta": {
    "company": "string|n/a",
    "title": "string|n/a"
  },
  "skills": {
    "required": [
      {"name": "string(lowercase)", "category": "language|framework|tool|concept|soft", "years": ">=Ny|n/a", "level": "junior|mid|senior|n/a"}
    ],
    "preferred": [
      {"name": "string(lowercase)", "category": "language|framework|tool|concept|soft", "years": ">=Ny|n/a", "level": "junior|mid|senior|n/a"}
    ]
  },
  "education": {
    "degrees": ["string"],
    "majors": ["string"]
  }
}
""".strip()

SYSTEM_PROMPT = f"""You are an information extractor.
STRICT RULES:
- Output **ONLY** JSON. No explanations, NO MARKDOWN CODE FENCES (EXRTEMELY IMPORTANT), no extra keys.
- Use the exact schema and field names specified below.
- Extract ONLY what appears in the JD text. Do not speculate or add missing items.
- Try your best to include all skills mentioned inside the JD and not to omit any item.
- Normalize skill names to lowercase; remove decorations (versions in parentheses). If years/level not explicit, use "n/a".
- Separate requirements into "required" vs "preferred" based on JD wording.

VOCABULARY (REFERENCE ONLY; NON-EXHAUSTIVE; NOT A WHITELIST):
- This list is only for canonicalization/synonyms.
- NEVER drop an item because it is not in the vocabulary.
- If a JD term ≈ a vocabulary term, use the vocabulary form.
- If not close enough, keep the JD term (normalized). DO NOT OMIT IT.
vocabulary lists (JSON; REFERENCE ONLY; DO NOT COPY INTO OUTPUT):
{vocab_json}

OUTPUT SCHEMA (strict; do not add or remove fields):
{SCHEMA_JSON}

SCOPE:
- Consider all parts of the JD (requirements/qualifications/responsibilities/role description/plus/preferred).
- Only include items truly mentioned in the JD.
"""

def build_prompt(jd_text: str) -> str:

    prompt = f"""JOB DESCRIPTION TEXT:
{jd_text}

REMEMBER **NOT** TO ADD MARKDOWN CODE FENCES LIKE:
```json
```
"""

    return prompt

# For debugging and an example to use
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from .llm_client import LLMClient
    from .storage import save_to_jsonl

    jsonl_path = "./test_output.jsonl"

    model = "gemini-2.5-pro"
    llm = LLMClient.init_gemini_client(model=model)
    llm.set_system_prompt(SYSTEM_PROMPT)

    job_description = """Lam Research logo
Lam Research
Share
Show more options
Systems Engineer 3 
Fremont, CA · Reposted 16 hours ago · Over 100 people clicked apply
Promoted by hirer · Responses managed off LinkedIn


$96K/yr - $215K/yr

Hybrid

Full-time

Apply

Save
Save Systems Engineer 3  at Lam Research
Systems Engineer 3
Lam Research · Fremont, CA (Hybrid)

Apply

Save
Save Systems Engineer 3  at Lam Research
Show more options
Your profile is missing required qualifications

Show match details

BETA

Is this information helpful?



Get personalized tips to stand out to hirers
Find jobs where you’re a top applicant and tailor your resume with the help of AI.

Try Premium for $0
People you can reach out to



Zhiyu (Alpha) and others in your network

Show all
About the job
The group you’ll be a part of

In the Global Products Group, we are dedicated to excellence in the design and engineering of Lam's etch and deposition products. We drive innovation to ensure our cutting-edge solutions are helping to solve the biggest challenges in the semiconductor industry.

The impact you’ll make

As a Systems Engineer at Lam, you'll design complex frameworks, systems, or products for our wafer fabrication equipment. You will define system specifications, requirements, parameters, and compatibility to integrate hardware, software, and firmware into a cohesive system. Identifying, analyzing, and resolving system design weaknesses, combined with your multi-layered technical expertise, you will help shape the next generation of Lam initiatives.

What You’ll Do

You will work as part of the Systems Engineering team in our engineering product development group to develop system and subsystem requirements, assist in project scope requests, define test plans, execute those test plans, analyze and present the test data, and manage system-level risks.

Your specific job responsibilities will include a subset of the following:

Identify and resolve technical opportunities and risks across different disciplines.
Create system specifications for subsystems of the transport module.
Manage design tradeoffs.
Design experiments to characterize subsystems.
Analyze data using Statistical Analysis of Variation and SPC methods.
Present test plans, test data, technical proposals, tradeoffs, etc.
Execute and deliver on tight schedules.
Perform hands-on lab work.

Who We’re Looking For

Minimum of 5 years of related experience with a Bachelor’s degree; or 3 years and a Master’s degree; or a PhD without experience; or equivalent work experience (candidates with less experience will be considered for a lower-level position).
Thrives in cross-functional teams. Systems engineers live in the intersection of hardware, controls, software, operators, manufacturing, etc. You need to be able to work well with all these teams.
Handles multiple tasks and priorities with attention to detail.

Preferred Qualifications

Coding experience; Python preferred.

Our commitment

We believe it is important for every person to feel valued, included, and empowered to achieve their full potential. By bringing unique individuals and viewpoints together, we achieve extraordinary results.

Lam Research ("Lam" or the "Company") is an equal opportunity employer. Lam is committed to and reaffirms support of equal opportunity in employment and non-discrimination in employment policies, practices and procedures on the basis of race, religious creed, color, national origin, ancestry, physical disability, mental disability, medical condition, genetic information, marital status, sex (including pregnancy, childbirth and related medical conditions), gender, gender identity, gender expression, age, sexual orientation, or military and veteran status or any other category protected by applicable federal, state, or local laws. It is the Company's intention to comply with all applicable laws and regulations. Company policy prohibits unlawful discrimination against applicants or employees.

Lam offers a variety of work location models based on the needs of each role. Our hybrid roles combine the benefits of on-site collaboration with colleagues and the flexibility to work remotely and fall into two categories – On-site Flex and Virtual Flex. ‘On-site Flex’ you’ll work 3+ days per week on-site at a Lam or customer/supplier location, with the opportunity to work remotely for the balance of the week. ‘Virtual Flex’ you’ll work 1-2 days per week on-site at a Lam or customer/supplier location, and remotely the rest of the time.

Salary

CA San Francisco Bay Area Salary Range for this position: $96,000.00 - $215,000.00.

The above salary range for this position is relevant to applicants that reside or work onsite in the California, San Francisco Bay Area only. Salary offers will depend on factors that include the location you work from, your level, education, training, specific skills, years of experience and comparison to other employees already in this role. Actual salary may vary from salary offered due to numerous factors including but not limited to unpaid time off, unpaid leave, company mandated shutdown, and other relevant factors.

Our Perks And Benefits

At Lam, our people make amazing things possible. That’s why we invest in you throughout the phases of your life with a comprehensive set of outstanding benefits.
"""

    prompt = build_prompt(job_description)
    response = llm.query(prompt)
    print(f"=== Response from AI ({model}) ===")
    print(response)

    save_to_jsonl(response, jsonl_path)