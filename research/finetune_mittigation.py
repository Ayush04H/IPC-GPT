import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
from sentence_transformers import SentenceTransformer, util

# Load LoRA fine-tuned model and tokenizer
model_path = "lora_model"  # Your model folder path
load_in_4bit = True  # Whether to load in 4-bit precision

model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if not load_in_4bit else torch.float32,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Load a Sentence Transformer for similarity checking
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

legal_corpus = {
    "IPC 187": "IPC Section 187 addresses the offense of harbouring an officer, soldier, sailor, or airman who has deserted from service. This section aims to prevent individuals from aiding and abetting those who have abandoned their duties to the armed forces.  The act of harbouring includes providing shelter, food, or any form of assistance to the deserter with the knowledge that they are a deserter. The punishment prescribed under IPC Section 187 is imprisonment which may extend to seven years, or a fine, or both.  The severity of the punishment often depends on the extent of the assistance provided and the circumstances surrounding the desertion.",

    "IPC 302": "IPC Section 302 deals with the crime of murder. It defines murder as the intentional killing of another human being, and it lays out the corresponding punishment. The punishment for murder, upon conviction, is either the death penalty (capital punishment) or life imprisonment.  In addition to either of these sentences, the convicted individual is also liable to pay a fine. The imposition of the death penalty is reserved for the 'rarest of rare' cases, as determined by the courts, considering the specific facts and circumstances of the crime, the background of the offender, and the impact on the victim's family.  Life imprisonment generally means imprisonment for the remainder of the convict's natural life, subject to possible remissions granted by the government.",

    "IPC 307": "IPC Section 307 pertains to the offense of attempt to murder. This section covers situations where a person intends to cause the death of another person and takes steps towards committing the act, but the act is not fully completed, and the intended victim survives. The punishment for attempt to murder under IPC Section 307 is imprisonment which may extend up to 10 years, along with a fine.  However, a significant element to note is that if the attempt to murder causes hurt to the victim, the imprisonment may extend to life imprisonment. This highlights the seriousness with which the law views attempts on human life, especially when bodily harm results from the attempt.",

    "IPC 375": "IPC Section 375 provides a comprehensive legal definition of the crime of rape. This section specifies the circumstances under which sexual intercourse constitutes rape.  According to IPC Section 375, a man is said to commit rape if he engages in sexual intercourse with a woman without her free and informed consent, or against her will, or without her consent at all, or when her consent has been obtained by putting her in fear of death or hurt.  Other circumstances defining rape include when the woman is incapable of giving consent due to unsoundness of mind or intoxication, or when she is under eighteen years of age. Section 375 essentially establishes the legal basis for determining whether a sexual act constitutes rape, while the punishment for the offense itself is prescribed under Section 376 of the IPC.",

    "IPC 376": "IPC Section 376 outlines the punishment for the crime of rape, as defined in IPC Section 375. The punishment prescribed under IPC Section 376 includes rigorous imprisonment, which shall not be less than ten years, but may extend to life imprisonment. In addition to the imprisonment, the convicted person is also liable to pay a fine. The severity of the punishment can vary depending on the circumstances of the offense, such as the age of the victim, the use of violence, and the relationship between the accused and the victim. In cases involving gang rape, the punishment can be even more severe, potentially leading to a death sentence in certain circumstances. The amendments to Section 376 reflect the increased recognition of the severity of sexual assault and the need for stringent penalties to deter such crimes.",

    "IPC 420": "IPC Section 420 addresses the offense of cheating and dishonestly inducing delivery of property. This section is frequently invoked in cases of fraud where individuals are deceived into parting with their money, goods, or valuable securities. To constitute an offense under IPC Section 420, there must be a deliberate intention to deceive the victim, and as a result of that deception, the victim must have been induced to deliver property or alter or destroy any valuable security. The punishment for committing an offense under IPC Section 420 is imprisonment for a term which may extend to seven years, and the offender is also liable to pay a fine. The severity of the punishment often depends on the value of the property involved and the degree of deception employed.",

    "IPC 498A": "IPC Section 498A specifically deals with cruelty towards a married woman by her husband or any relative of her husband. This section was introduced to address the rising cases of dowry harassment and other forms of cruelty inflicted upon women within their marital homes.  'Cruelty' under this section is broadly defined and includes any conduct that is likely to drive the woman to commit suicide or cause grave injury to her health (mental or physical), or harassment with a view to coercing her or any person related to her to meet any unlawful demand for any property or valuable security.  The punishment for an offense under IPC Section 498A is imprisonment for a term which may extend to three years, and the offender is also liable to pay a fine.  This section is often subject to scrutiny due to its potential misuse, and courts have issued guidelines to prevent false accusations and harassment.",

    "IPC 506": "IPC Section 506 covers the offense of criminal intimidation. This section applies when someone threatens another person with injury to their person, reputation, or property, or to the person or reputation of anyone in whom that person is interested, with the intent to cause alarm to that person, or to cause that person to do any act which he is not legally bound to do, or to omit to do any act which that person is legally entitled to do, as the means of avoiding the execution of such threat. The punishment for criminal intimidation varies depending on the severity of the threat. If the threat is to cause death or grievous hurt, or to cause the destruction of any property by fire, or to commit an offense punishable with death or imprisonment for life, or to impute unchastity to a woman, the imprisonment may extend to seven years, or a fine, or both. In other cases of criminal intimidation, the punishment may be imprisonment for a term which may extend to two years, or a fine, or both.",

    "IPC 509": "IPC Section 509 addresses the offense of insulting the modesty of a woman through words, gesture, or act. This section aims to protect women from any form of harassment or behavior that is intended to outrage their modesty.  The key element of this offense is the intention to insult the modesty of a woman.  The act must be such that it would reasonably offend the sense of decency and propriety of a woman.  The punishment for committing an offense under IPC Section 509 is imprisonment for a term which may extend to three years, and the offender is also liable to pay a fine. This section is often invoked in cases of street harassment, inappropriate comments, or gestures that are directed towards women.",

    "IPC 511": "IPC Section 511 deals with the offense of attempting to commit crimes that are punishable with imprisonment for life or other imprisonment. This section is applicable when a person attempts to commit an offense but fails to complete it. The punishment prescribed under IPC Section 511 is generally half of the punishment that is provided for the completed offense.  For instance, if the offense attempted is punishable with life imprisonment, the punishment for the attempt may be imprisonment for up to half of the term.  IPC Section 511 is crucial in holding individuals accountable for their criminal intentions and actions, even if the intended crime is not ultimately carried out. The determination of whether an act constitutes an 'attempt' involves assessing the proximity of the act to the completed offense and the intention of the accused.",

    "IPC 34": "IPC Section 34 embodies the principle of 'common intention' in criminal law. This section provides that when a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone.  This means that if a group of individuals jointly commits a crime with a shared intention, each member of the group is equally responsible for the entire offense, regardless of the specific role they played.  The essence of Section 34 lies in the pre-arranged plan and the participation of all individuals in the act with the common goal of achieving a specific criminal outcome.  It is not necessary for each member to perform every part of the act; it is sufficient if they are present and participate in the commission of the offense with the shared intention.",

    "IPC 120B": "IPC Section 120B deals with the offense of criminal conspiracy. This section is invoked when two or more persons agree to do, or cause to be done, an illegal act, or an act which is not illegal by illegal means.  The agreement itself constitutes the offense of criminal conspiracy, even if the agreed-upon act is never carried out. The punishment for criminal conspiracy depends on the nature of the offense that was the object of the conspiracy. If the conspiracy is to commit an offense punishable with death, imprisonment for life, or rigorous imprisonment for a term of two years or upwards, the punishment is the same as if the conspirator had abetted such an offense.  In other cases of criminal conspiracy, the punishment may be imprisonment for a term not exceeding six months, or a fine, or both.  The key element of criminal conspiracy is the agreement to commit an illegal act, and the prosecution must prove the existence of this agreement beyond a reasonable doubt.",

    "IPC 144": "IPC Section 144 empowers authorities to impose restrictions on public gatherings in order to prevent potential disturbances of public order. This section authorizes a District Magistrate, a Sub-divisional Magistrate, or any other Executive Magistrate specially empowered by the State Government to issue an order prohibiting the assembly of four or more persons in an area. This section is often invoked to prevent riots, public disorder, or any activity that could disrupt peace and tranquility. The order can specify the area where the restrictions apply, the duration of the restrictions, and the activities that are prohibited.  Violating an order issued under IPC Section 144 can result in arrest and prosecution.  This section is frequently used to impose curfews or restrict movement during times of unrest or potential social disruption.  The exercise of powers under Section 144 is subject to judicial review to ensure that it is not used arbitrarily or excessively.",

    "IPC 153A": "IPC Section 153A addresses the offense of promoting enmity between different groups on grounds of religion, race, place of birth, residence, language, etc., and doing acts prejudicial to maintenance of harmony. This section aims to curb activities that incite hatred, ill-will, or disharmony between different communities, thereby preserving social peace and cohesion. To constitute an offense under IPC Section 153A, there must be an intention to promote feelings of enmity, hatred, or ill-will between different groups, or an act that is prejudicial to the maintenance of harmony between them.  This section often involves the use of inflammatory speeches, writings, or visual representations that are likely to incite violence or hatred. The punishment for an offense under IPC Section 153A is imprisonment which may extend to three years, or a fine, or both. This section is a critical tool in preventing communal violence and promoting a harmonious society, but its application is often debated in the context of freedom of speech and expression.",

    "IPC 268": "IPC Section 268 defines the offense of public nuisance. A person is guilty of a public nuisance who does any act or is guilty of an illegal omission which causes any common injury, danger or annoyance to the public or to the people in general who dwell or occupy property in the vicinity, or which must necessarily cause injury, obstruction, danger or annoyance to persons who may have occasion to use any public right. This section is broadly construed and can cover a wide range of activities that affect the health, safety, convenience, or comfort of the public. Examples of public nuisance include obstructing a public street, creating excessive noise, or polluting the environment. Whether a particular act constitutes a public nuisance is determined by the nature of the act, its impact on the public, and the reasonableness of the conduct. While IPC 268 defines the offense, the punishment for it is typically outlined in other related sections of the IPC or other relevant laws.  The goal is to maintain public order and protect the common good.",

    "IPC 295A": "IPC Section 295A addresses deliberate and malicious acts intended to outrage religious feelings of any class by insulting its religion or religious beliefs. This section is aimed at protecting the religious sentiments of all communities and preventing acts that are likely to cause religious disharmony or violence. To constitute an offense under IPC Section 295A, the act must be deliberate and malicious, meaning that it is done with the intention to insult the religion or religious beliefs of a particular group. The act must also be of such a nature that it is likely to outrage the religious feelings of that group. This section often involves the use of offensive language, images, or symbols that are considered blasphemous or sacrilegious. The punishment for an offense under IPC Section 295A is imprisonment which may extend to three years, or a fine, or both. Like Section 153A, this section is important for maintaining religious harmony, but its application can be controversial due to the potential for it to be used to stifle legitimate criticism or dissent.",

}


# Function to check similarity with legal corpus
def check_similarity(response, corpus):
    corpus_texts = list(corpus.values())
    corpus_embeddings = similarity_model.encode(corpus_texts, convert_to_tensor=True)
    response_embedding = similarity_model.encode(response, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(response_embedding, corpus_embeddings)
    max_similarity, index = torch.max(similarities, dim=1)
    
    return max_similarity.item(), list(corpus.keys())[index] if max_similarity.item() > 0.7 else None

# Function to generate response and evaluate hallucination
def evaluate_hallucination(query):
    messages = [{"role": "user", "content": query}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            min_p=0.1
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Check factual accuracy
    similarity_score, matched_section = check_similarity(response, legal_corpus)

    # **Print Results in a Detailed Manner**
    print("\n" + "="*50)
    print("🔎 **User Query:**")
    print(query)
    print("="*50)

    print("\n📌 **Model's Generated Response:**")
    print(response)
    print("="*50)

    if matched_section:
        print("✅ **Factually Correct**")
        print(f"📖 Matched Legal Section: {matched_section}")
        print(f"🔢 Similarity Score: {similarity_score:.2f}")
    else:
        print("⚠️ **Possible Hallucination Detected!**")
        print("⚠️ No reliable legal reference found in the corpus.")
        print(f"🔢 Similarity Score: {similarity_score:.2f} (Below Threshold)")

    print("="*50)

# Example Evaluation
query = "How does IPC 34 affect shared criminal responsibility?"
evaluate_hallucination(query)
