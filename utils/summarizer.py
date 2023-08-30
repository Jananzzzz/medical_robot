from transformers import PegasusForConditionalGeneration
# Need to download tokenizers_pegasus.py and other Python script from Fengshenbang-LM github repo in advance,
# or you can download tokenizers_pegasus.py and data_utils.py in https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M/tree/main
# Strongly recommend you git clone the Fengshenbang-LM repo:
# 1. git clone https://github.com/IDEA-CCNL/Fengshenbang-LM
# 2. cd Fengshenbang-LM/fengshen/examples/pegasus/
# and then you will see the tokenizers_pegasus.py and data_utils.py which are needed by pegasus model
# import sys
# sys.path.append("/Fengshenbang_LM/fengshen/examples/pegasus/")
# print(sys.path)
from Fengshenbang_LM.fengshen.examples.pegasus.tokenizers_pegasus import PegasusTokenizer


model = PegasusForConditionalGeneration.from_pretrained("/Users/janan/Chinese-medical-dialogue-data/Randeng-Pegasus-523M-Summary-Chinese")
tokenizer = PegasusTokenizer.from_pretrained("/Users/janan/Chinese-medical-dialogue-data/Randeng-Pegasus-523M-Summary-Chinese")

text = "据微信公众号“界面”报道，4日上午10点左右，中国发改委反垄断调查小组突击查访奔驰上海办事处，调取数据材料，并对多名奔驰高管进行了约谈。截止昨日晚9点，包括北京梅赛德斯-奔驰销售服务有限公司东区总经理在内的多名管理人员仍留在上海办公室内"
text = """
你的情况最好是还是采取人工周期的方式实施救治，否则的话会致使不育的另外也会干扰正常的夫妻生活的，致使阴道干涩的情况建议你还是积极的实施救治，留意歇息也可以中西医结合实施救治超越转好卵巢早衰的治疗方法有许多，但是由于患者病情不同所以采用的治疗方法也就不一样，因此建议患者发觉症状后，及早实施诊断救治。
是内分泌紊乱导致的，可以服用调经丸进行治疗.效果不错的建议尽量使你的生活有规律，防止受寒，无论何时都要避免受寒，多吃含有铁和滋补性的食物，月经不调的出现不仅仅会影响女性的健康，还会导致妇科疾病的出现，因此，最好去医院的妇科做一下B超检查，找出病因后对症治疗，平时应该注意多喝水，注意个人卫生。
1、保持精神愉快，避免精神刺激和情绪波动。个别女性在月经期有下腹发胀、腰酸、乳房胀痛、轻度腹泻、容易疲倦、嗜睡、情绪不稳定、易怒或易忧郁等现象，均属正常，不必过分紧张。
2、注意卫生，预防感染。注意外生殖器的卫生清洁。注意保暖，避免寒冷刺激。避免过劳。经血量多者忌食红糖。
3、防止过度节食，戒烟限酒，注意自己的饮食结构，多食用瘦肉、谷类、深绿叶蔬菜及含钙丰富的食物，不宜过食生冷，保持心情舒畅，加强锻炼，提高身体素质。
4、注意内裤要柔软、棉质，通风透气性能良好，要勤洗勤换，换洗的内裤要放在阳光下晒干。
5、不宜吃生冷、酸辣等刺激性食物，多饮开水，保持大便通畅。血热者经期前宜多食新鲜水果和蔬菜，忌食葱蒜韭姜等刺激运火之物。气血虚者平时必须增加营养，如牛奶、鸡蛋、豆浆、猪肝、菠菜、猪肉、鸡肉、羊肉等，忌食生冷瓜果。
6、长期月经不调要注意饮食调理，如多喝黑木耳红枣茶、浓茶红糖饮、山楂红糖饮等。如果症状仍未改善建议及时到医院检查，明确病因，以便遵医嘱及早针对性的治疗。
"""
text = "你好：小儿发生感冒时，做家长的一定要照医嘱做好家庭护理。小儿感冒家庭护理重要的一条是要让孩子充分休息，病儿年龄越小，越是需要休息，待症状消失后才能恢复自由活动。其二是按时服药。就大多数感冒而言，多数是由于病毒所致，抗菌药物无效，特别是早期病毒感染，抗生素非但无效，滥用抗生素反而会引起机体菌群失调，有利病菌繁殖，加重病情。服用“百服咛”能较好地解除感冒引起发热、鼻塞、咳嗽等不适，避免并发症发生，及早康复。其三，小儿感冒发热期，应根据孩子食欲及消化能力不同，分别给予流质或面条，稀粥等食物。喂奶的孩子应暂时减少次数,以免发生吐泻等消化不良症状。其四，居室安静，空气新鲜，禁烟，温度宜恒定，不要太高，或太低、太湿，有喉炎症状时更应注意，这样才能让患儿早早康复。"
inputs = tokenizer(text, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
print("start generating summary")
# print out the summary
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# model Output: 反垄断调查小组突击查访奔驰上海办事处，对多名奔驰高管进行约谈

