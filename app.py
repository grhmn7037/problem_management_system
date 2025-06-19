from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from markupsafe import Markup, escape
import re

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as SumySummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# from collections import Counter # ليست ضرورية الآن مع get_suggested_root_causes الحالية

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here_V9_ROOT_CAUSE_FIXED'  # تحديث المفتاح

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///problem_management.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

analyzer = SentimentIntensityAnalyzer()
rake_nltk_var = Rake()


def nl2br_filter(s):
    if s:
        s_escaped = escape(str(s))
        return Markup(re.sub(r'(\r\n|\n|\r)', '<br>\n', s_escaped))
    return ''


app.jinja_env.filters['nl2br'] = nl2br_filter


# --- نماذج قاعدة البيانات ---
# ... (جميع نماذجك تبقى كما هي تمامًا) ...
class Problem(db.Model):  # مثال مختصر
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description_initial = db.Column(db.Text, nullable=True)
    domain = db.Column(db.String(100), nullable=True)
    complexity_level = db.Column(db.String(50), nullable=True)
    date_identified = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    date_closed = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(50), default='مفتوحة')
    stakeholders_involved = db.Column(db.Text, nullable=True)
    initial_impact_assessment = db.Column(db.Text, nullable=True)
    problem_source = db.Column(db.String(150), nullable=True)
    refined_problem_statement_final = db.Column(db.Text, nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    sentiment_label = db.Column(db.String(50), nullable=True)
    problem_tags = db.Column(db.Text, nullable=True)
    ai_generated_summary = db.Column(db.Text, nullable=True)
    understanding_details = db.relationship('ProblemUnderstanding', backref='problem_parent', uselist=False,
                                            cascade="all, delete-orphan")
    cause_analysis_details = db.relationship('CauseAnalysis', backref='problem_parent', uselist=False,
                                             cascade="all, delete-orphan")
    proposed_solutions = db.relationship('ProposedSolution', backref='problem_parent', lazy='dynamic',
                                         cascade="all, delete-orphan")
    chosen_solutions = db.relationship('ChosenSolution', backref='problem_parent', lazy='dynamic',
                                       cascade="all, delete-orphan")
    lessons_learned = db.relationship('LessonLearned', backref='problem_parent', uselist=False,
                                      cascade="all, delete-orphan")

    def __repr__(self): return f'<Problem {self.title}>'


class ProblemUnderstanding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.Integer, db.ForeignKey('problem.id', ondelete='CASCADE'), nullable=False)
    active_listening_notes = db.Column(db.Text, nullable=True)
    key_questions_asked = db.Column(db.Text, nullable=True)
    initial_data_sources = db.Column(db.Text, nullable=True)
    initial_hypotheses = db.Column(db.Text, nullable=True)
    stakeholder_feedback_initial = db.Column(db.Text, nullable=True)
    refined_problem_statement_early = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class CauseAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.Integer, db.ForeignKey('problem.id', ondelete='CASCADE'), nullable=False)
    data_collection_methods_deep = db.Column(db.Text, nullable=True)
    data_analysis_techniques_used = db.Column(db.Text, nullable=True)
    key_findings_from_analysis = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    potential_root_causes = db.relationship('PotentialRootCause', backref='analysis_parent', lazy='dynamic',
                                            cascade="all, delete-orphan")


class PotentialRootCause(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('cause_analysis.id', ondelete='CASCADE'), nullable=False)
    cause_description = db.Column(db.Text, nullable=False)
    evidence_supporting_cause = db.Column(db.Text, nullable=True)
    validation_status = db.Column(db.String(50), default='محتمل')
    impact_of_cause = db.Column(db.Text, nullable=True)


class ProposedSolution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.Integer, db.ForeignKey('problem.id', ondelete='CASCADE'), nullable=False)
    solution_description = db.Column(db.Text, nullable=False)
    generation_method = db.Column(db.Text, nullable=True)
    estimated_cost = db.Column(db.String(100), nullable=True)
    estimated_time_to_implement = db.Column(db.String(100), nullable=True)
    potential_benefits = db.Column(db.Text, nullable=True)
    potential_risks = db.Column(db.Text, nullable=True)
    is_chosen = db.Column(db.Boolean, default=False)


class SolutionEvaluationCriterion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)


class ChosenSolution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.Integer, db.ForeignKey('problem.id', ondelete='CASCADE'), nullable=False)
    proposed_solution_id = db.Column(db.Integer, db.ForeignKey('proposed_solution.id', ondelete='CASCADE'),
                                     nullable=False)
    justification_for_choice = db.Column(db.Text, nullable=True)
    approval_status = db.Column(db.String(50), nullable=True)
    date_chosen = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    proposed_solution_details = db.relationship('ProposedSolution', backref=db.backref('chosen_link', uselist=False))
    implementation_plan = db.relationship('ImplementationPlan', backref='chosen_solution_parent', uselist=False,
                                          cascade="all, delete-orphan")
    kpis = db.relationship('SolutionKPI', backref='chosen_solution_parent', lazy='dynamic',
                           cascade="all, delete-orphan")


class ImplementationPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chosen_solution_id = db.Column(db.Integer, db.ForeignKey('chosen_solution.id', ondelete='CASCADE'), nullable=False)
    plan_description = db.Column(db.Text, nullable=True)
    overall_status = db.Column(db.String(50), default='لم يبدأ')
    start_date_planned = db.Column(db.DateTime, nullable=True)
    end_date_planned = db.Column(db.DateTime, nullable=True)
    start_date_actual = db.Column(db.DateTime, nullable=True)
    end_date_actual = db.Column(db.DateTime, nullable=True)
    overall_budget = db.Column(db.String(100), nullable=True)
    key_personnel = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    tasks = db.relationship('ImplementationTask', backref='plan_parent', lazy='dynamic', cascade="all, delete-orphan")


class ImplementationTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('implementation_plan.id', ondelete='CASCADE'), nullable=False)
    task_description = db.Column(db.Text, nullable=False)
    assigned_to = db.Column(db.String(150), nullable=True)
    priority = db.Column(db.String(50), default='متوسط')
    task_status = db.Column(db.String(50), default='لم تبدأ')
    due_date = db.Column(db.DateTime, nullable=True)
    completion_date = db.Column(db.DateTime, nullable=True)
    notes = db.Column(db.Text, nullable=True)


class SolutionKPI(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chosen_solution_id = db.Column(db.Integer, db.ForeignKey('chosen_solution.id', ondelete='CASCADE'), nullable=False)
    kpi_name = db.Column(db.String(200), nullable=False)
    kpi_description = db.Column(db.Text, nullable=True)
    target_value = db.Column(db.String(100), nullable=True)
    current_value_baseline = db.Column(db.String(100), nullable=True)
    measurement_unit = db.Column(db.String(50), nullable=True)
    measurement_frequency = db.Column(db.String(50), nullable=True)
    measurements = db.relationship('KPIMeasurement', backref='kpi_parent', lazy='dynamic', cascade="all, delete-orphan")


class KPIMeasurement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kpi_id = db.Column(db.Integer, db.ForeignKey('solution_kpi.id', ondelete='CASCADE'), nullable=False)
    measurement_date = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    actual_value = db.Column(db.String(100), nullable=False)
    notes = db.Column(db.Text, nullable=True)


class LessonLearned(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    problem_id = db.Column(db.Integer, db.ForeignKey('problem.id', ondelete='CASCADE'), nullable=False)
    what_went_well = db.Column(db.Text, nullable=True)
    what_could_be_improved = db.Column(db.Text, nullable=True)
    recommendations_for_future = db.Column(db.Text, nullable=True)
    key_takeaways = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


# --- دوال مساعدة ---
def calculate_sentiment(text):  # ...
    if not text or not text.strip(): return None, None
    vs = analyzer.polarity_scores(text)
    score = vs['compound']
    if score >= 0.05:
        label = "إيجابي"
    elif score <= -0.05:
        label = "سلبي"
    else:
        label = "محايد"
    return score, label


def extract_keywords(text, max_keywords=5):  # ...
    if not text or not text.strip(): return []
    try:
        rake_nltk_var.extract_keywords_from_text(text)
        ranked_phrases_with_scores = rake_nltk_var.get_ranked_phrases_with_scores()
        keywords = [phrase for score, phrase in ranked_phrases_with_scores[:max_keywords]]
        print(f"DEBUG: RakeNLTK extracted phrases for '{text[:50]}...': {keywords}")
        return keywords
    except Exception as e:
        print(f"Error in extract_keywords with RakeNLTK: {e}")
        return []


def generate_ai_summary(text, language="english", sentences_count=2):  # ...
    MIN_WORDS_FOR_SUMMARIZATION = 30
    if not text or not text.strip() or len(text.split()) < MIN_WORDS_FOR_SUMMARIZATION:
        print(f"DEBUG: Text too short to summarize or empty: '{text[:50]}...'")
        return None
    try:
        if language == "auto_detect":
            if any('\u0600' <= char <= '\u06FF' for char in text[:100]):
                language = "arabic"
                print(
                    f"DEBUG: Auto-detected language as Arabic for summary. Sumy support for Arabic stopwords/stemmer is limited.")
            else:
                language = "english"
                print(f"DEBUG: Auto-detected language as non-Arabic, defaulting to English for summary.")
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer_to_use = None
        try:
            stemmer_to_use = Stemmer(language)
        except ValueError:
            print(f"DEBUG: Sumy Stemmer not available for language '{language}'.")
        summarizer = SumySummarizer(stemmer_to_use) if stemmer_to_use else SumySummarizer()
        try:
            summarizer.stop_words = get_stop_words(language)
        except OSError:
            print(f"DEBUG: Sumy Stopwords not available for language '{language}'.")
            summarizer.stop_words = []
        summary_sentences_obj = summarizer(parser.document, sentences_count)
        summary_sentences = [str(sentence) for sentence in summary_sentences_obj]
        summary = " ".join(summary_sentences)
        if not summary.strip():
            print(f"DEBUG: Sumy returned empty summary for '{text[:50]}...'. Taking first sentences as fallback.")
            sentences = re.split(r'(?<=[.!?؟])\s+', text.strip())
            if sentences:
                alternative_summary = " ".join(sentences[:sentences_count])
                max_fallback_words = sentences_count * 20
                if len(alternative_summary.split()) > max_fallback_words:
                    alternative_summary = " ".join(alternative_summary.split()[:sentences_count * 15]) + "..."
                summary = alternative_summary
            else:
                summary = " ".join(text.split()[:sentences_count * 10]) + "..."
        print(f"DEBUG: Sumy generated summary: {summary[:100]}...")
        return summary.strip() if summary else None
    except Exception as e:
        print(f"Error generating summary with Sumy: {e}")
        try:
            fallback_summary = " ".join(text.split()[:sentences_count * 10]) + "..."
            print(f"DEBUG: Fallback summary (due to Sumy error): {fallback_summary[:100]}...")
            return fallback_summary
        except:
            return None


def parse_date_string(date_str, field_name_for_flash="التاريخ", date_format='%Y-%m-%d'):  # ...
    if date_str and date_str.strip():
        try:
            return datetime.datetime.strptime(date_str, date_format)
        except ValueError:
            flash(f"تنسيق {field_name_for_flash} '{date_str}' غير صحيح (المتوقع: {date_format}). تم تجاهله.", "warning")
            return None
    return None


# <<< بداية: قاعدة المعرفة ودالة اقتراح الأسباب الجذرية (تمت إضافتها هنا) >>>
SUGGESTED_ROOT_CAUSES_KNOWLEDGE_BASE = {
    "بطء": ["نقص موارد الخادم (CPU/RAM)", "مشكلة في أداء الشبكة", "استعلام قاعدة بيانات غير مُحسَّن",
            "فهرسة غير كافية في قاعدة البيانات"],
    "توقف": ["خطأ برمجي غير متوقع", "نفاد ذاكرة التطبيق/الخادم", "فشل في مكون حاسم", "مشكلة في مصدر الطاقة"],
    "خطأ": ["إدخال بيانات غير صالح من المستخدم", "مشكلة توافق بين الأنظمة/المكونات", "خطأ في منطق البرنامج",
            "بيانات تالفة"],
    "سيرفر": ["نقص موارد الخادم (CPU/RAM/Disk)", "مشكلة في نظام تشغيل الخادم", "فشل في القرص الصلب للخادم",
              "إعدادات خاطئة في الخادم"],
    "شبكة": ["مشكلة في كابل الشبكة أو المحول", "ازدحام شديد في الشبكة", "إعدادات جدار الحماية تمنع الاتصال",
             "مشكلة في خدمة DNS"],
    "database": ["استعلام قاعدة بيانات غير مُحسَّن", "فهرسة غير كافية", "قفل في قاعدة البيانات"],
    "login": ["كلمة مرور خاطئة", "الحساب مقفل", "مشكلة في خدمة المصادقة"],
    "بطئ": ["نقص موارد الخادم (CPU/RAM)", "مشكلة في أداء الشبكة", "استعلام قاعدة بيانات غير مُحسَّن"],
    "تطبيق لا يستجيب": ["نفاد ذاكرة التطبيق", "حلقة لا نهائية في الكود", "قفل (Deadlock) بين العمليات"],
    "بيانات خاطئة": ["خطأ في إدخال البيانات", "خطأ في عملية تحويل/معالجة البيانات", "مشكلة في مصدر البيانات"],
    "عدم القدرة على الوصول": ["الخدمة متوقفة", "مشكلة في الشبكة أو الاتصال بالإنترنت", "عنوان URL خاطئ"],
    "انقطاع": ["انقطاع التيار الكهربائي", "فشل في الاتصال بالشبكة", "توقف خدمة أساسية"],
    "slow": ["Server resource shortage", "Network performance issue", "Inefficient database query"],
    "crash": ["Unhandled software error", "Application/Server out of memory", "Critical component failure"],
    "error": ["Invalid user input", "Compatibility issue", "Software logic error"]
}


def get_suggested_root_causes(problem_description, problem_tags_str):
    suggested_causes = set()
    text_to_search_in = ""
    if problem_description:
        text_to_search_in += problem_description.lower() + " "
    if problem_tags_str:
        tags = [tag.strip().lower() for tag in problem_tags_str.split(',')]
        text_to_search_in += " ".join(tags)

    if not text_to_search_in.strip():
        return []

    print(f"DEBUG: Text for Root Cause Suggestion: {text_to_search_in[:200]}...")
    for keyword, causes in SUGGESTED_ROOT_CAUSES_KNOWLEDGE_BASE.items():
        # البحث عن الكلمة المفتاحية ككلمة كاملة أو كجزء من النص
        # استخدام re.escape ضروري إذا كانت الكلمات المفتاحية قد تحتوي على رموز خاصة
        # pattern = r'\b' + re.escape(keyword.lower()) + r'\b' # للبحث عن كلمة كاملة
        # بحث أبسط حاليًا:
        if keyword.lower() in text_to_search_in:
            print(f"DEBUG: Found keyword '{keyword}' for root cause suggestion.")
            for cause in causes:
                suggested_causes.add(cause)
    return list(suggested_causes)


# <<< بداية: قاعدة المعرفة للـ Chatbot وإعداداته >>>
FAQ_KNOWLEDGE_BASE = {
    "كيفية إضافة مشكلة": "لإضافة مشكلة جديدة، انقر على زر 'إضافة مشكلة جديدة' في شريط التنقل العلوي أو في الصفحة الرئيسية، ثم قم بملء الحقول المطلوبة واضغط على 'حفظ المشكلة'.",
    "اضافة مشكلة": "لإضافة مشكلة جديدة، انقر على زر 'إضافة مشكلة جديدة' في شريط التنقل العلوي أو في الصفحة الرئيسية، ثم قم بملء الحقول المطلوبة واضغط على 'حفظ المشكلة'.",
    "ما هي حالة المشكلة": "حالة المشكلة تشير إلى المرحلة الحالية التي تمر بها المشكلة، مثل 'مفتوحة'، 'قيد التحليل'، 'مغلقة'، إلخ. يمكنك تعديل الحالة من صفحة تعديل المشكلة.",
    "حالة المشكلة": "حالة المشكلة تشير إلى المرحلة الحالية التي تمر بها المشكلة، مثل 'مفتوحة'، 'قيد التحليل'، 'مغلقة'، إلخ. يمكنك تعديل الحالة من صفحة تعديل المشكلة.",
    "تعديل مشكلة": "لتعديل مشكلة، اذهب إلى تفاصيل المشكلة ثم انقر على زر 'تعديل بيانات المشكلة الرئيسية'، أو من القائمة الرئيسية انقر على زر 'تعديل' بجانب المشكلة.",
    "كيف اعدل وصف": "لتعديل وصف المشكلة، اذهب إلى صفحة تعديل المشكلة وقم بتغيير النص في حقل 'الوصف الأولي للمشكلة' ثم احفظ التعديلات.",
    "ما هي الكلمات المفتاحية": "الكلمات المفتاحية تساعد في تصنيف المشكلة وتسهيل البحث عنها. يمكنك إدخالها يدويًا أو سيقوم النظام باقتراح بعضها بناءً على وصف المشكلة. يمكنك أيضًا إضافة كلمات مفتاحية تفاعليًا عند كتابة الوصف.",
    "تلخيص": "يقوم النظام بإنشاء ملخص تلقائي للوصف الأولي للمشكلة إذا كان طويلاً للمساعدة في فهمها بسرعة. يظهر هذا الملخص في صفحة تفاصيل المشكلة.",
    "مشاعر": "يقوم النظام بتحليل المشاعر الكامنة في الوصف الأولي للمشكلة (إيجابية، سلبية، محايدة) ويعرضها في صفحة التفاصيل وفي القائمة الرئيسية.",
    "مساعدة": "أنا هنا لمساعدتك! يمكنك أن تسألني عن: \n- كيفية إضافة مشكلة \n- كيفية تعديل مشكلة \n- ما هي حالة المشكلة \n- ما هي الكلمات المفتاحية \n- ما هو التلخيص التلقائي \n- ما هو تحليل المشاعر",
    "شكرا": "على الرحب والسعة! سعيد بمساعدتك.",
    "help": "أنا هنا لمساعدتك! يمكنك أن تسألني عن: \n- كيفية إضافة مشكلة \n- كيفية تعديل مشكلة \n- ما هي حالة المشكلة \n- ما هي الكلمات المفتاحية \n- ما هو التلخيص التلقائي \n- ما هو تحليل المشاعر",
    "": "أنا هنا لمساعدتك! يمكنك أن تسألني عن: \n- كيفية إضافة مشكلة \n- كيفية تعديل مشكلة \n- ما هي حالة المشكلة \n- ما هي الكلمات المفتاحية \n- ما هو التلخيص التلقائي \n- ما هو تحليل المشاعر",

    "اهلا": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟",
    "مرحبا": "مرحباً بك! كيف يمكنني مساعدتك اليوم؟",
    "سلام عليكم": "مرحباً بك! كيف يمكنني مساعدتك اليوم؟",
    "اهلا": "مرحباً بك! كيف يمكنني مساعدتك اليوم؟",
    "الكلمات المفتاحية": "الكلمات المفتاحية تساعد في تصنيف المشكلة وتسهيل البحث عنها. يمكنك إدخالها يدويًا أو سيقوم النظام باقتراح بعضها بناءً على وصف المشكلة. يمكنك أيضًا إضافة كلمات مفتاحية تفاعليًا عند كتابة الوصف.",
    "إضافة مشكلة": "لإضافة مشكلة جديدة، انقر على زر 'إضافة مشكلة جديدة' في شريط التنقل العلوي أو في الصفحة الرئيسية، ثم قم بملء الحقول المطلوبة واضغط على 'حفظ المشكلة'.",
    "كيف حالك": "مرحباً بك! كيف يمكنني مساعدتك اليوم؟"
    # يمكنك إضافة المزيد من الكلمات المفتاحية والأسئلة التي تغطي وظائف النظام
}
DEFAULT_CHATBOT_RESPONSE = "عذرًا، لم أفهم سؤالك. هل يمكنك إعادة صياغته أو تجربة سؤال آخر؟ اكتب 'مساعدة' لعرض قائمة بالمواضيع التي يمكنني المساعدة بها."


# دالة بسيطة لمطابقة السؤال مع قاعدة المعرفة
def get_chatbot_response(user_message):
    user_message_lower = user_message.lower().strip()

    # بحث عن تطابق مباشر للمفاتيح (التي هي أسئلة أو كلمات دالة)
    for keyword, response in FAQ_KNOWLEDGE_BASE.items():
        if keyword.lower() in user_message_lower:  # إذا كانت الكلمة المفتاحية موجودة في رسالة المستخدم
            return response

    # إذا لم يتم العثور على تطابق مباشر، يمكن إضافة منطق أكثر تعقيدًا هنا لاحقًا
    # مثل البحث عن كلمات متعددة أو استخدام تقنيات تشابه النصوص البسيطة.
    # حاليًا، سنرجع الرد الافتراضي.
    return DEFAULT_CHATBOT_RESPONSE


# <<< نهاية: قاعدة المعرفة للـ Chatbot وإعداداته >>>

# <<< نهاية: قاعدة المعرفة ودالة اقتراح الأسباب الجذرية >>>

# --- المسارات (Routes) ---
@app.route('/')
def index():  # ...
    all_problems = Problem.query.order_by(Problem.date_identified.desc()).all()
    return render_template('index.html', problems=all_problems)


@app.route('/add_problem', methods=['GET', 'POST'])
def add_problem():  # ...
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        title = request.form['title']
        description_initial = request.form.get('description_initial')
        domain = request.form.get('domain')
        complexity_level = request.form.get('complexity_level')
        status = request.form.get('status', 'مفتوحة')
        stakeholders_involved = request.form.get('stakeholders_involved')
        initial_impact_assessment = request.form.get('initial_impact_assessment')
        problem_source = request.form.get('problem_source')
        problem_tags_input = request.form.get('problem_tags', '')
        sentiment_score_value, sentiment_label_value = calculate_sentiment(description_initial)
        final_tags_to_save = problem_tags_input
        if not final_tags_to_save and description_initial:
            suggested_keywords_list = extract_keywords(description_initial)
            final_tags_to_save = ", ".join(suggested_keywords_list)
            if not problem_tags_input:
                form_data_to_pass['suggested_tags_on_error'] = suggested_keywords_list
        summary_text = None
        if description_initial:
            summary_text = generate_ai_summary(description_initial, language="auto_detect", sentences_count=2)
        new_problem = Problem(title=title, description_initial=description_initial, domain=domain,
                              complexity_level=complexity_level, status=status,
                              stakeholders_involved=stakeholders_involved,
                              initial_impact_assessment=initial_impact_assessment, problem_source=problem_source,
                              sentiment_score=sentiment_score_value, sentiment_label=sentiment_label_value,
                              problem_tags=final_tags_to_save, ai_generated_summary=summary_text)
        try:
            db.session.add(new_problem)
            db.session.commit()
            flash(f"تمت إضافة المشكلة '{title}' بنجاح!", 'success')
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء إضافة المشكلة: {str(e)}", 'danger')
            return render_template('add_problem.html', form_data=form_data_to_pass)
    return render_template('add_problem.html', form_data=form_data_to_pass)


def add_default_criteria_if_empty():  # ...
    if not SolutionEvaluationCriterion.query.first():
        default_criteria = [
            {"name": "التكلفة", "description": "التكلفة المالية لتنفيذ الحل."},
            {"name": "الوقت للتنفيذ", "description": "الوقت المتوقع لتطبيق الحل بالكامل."},
            {"name": "الجدوى الفنية", "description": "مدى سهولة أو صعوبة تطبيق الحل من الناحية التقنية."},
            {"name": "الأثر المتوقع", "description": "مدى فعالية الحل في معالجة المشكلة أو أسبابها الجذرية."},
            {"name": "المخاطر المحتملة", "description": "المخاطر أو العيوب المحتملة المصاحبة للحل."},
            {"name": "قبول أصحاب المصلحة", "description": "مدى تقبل ودعم الأطراف المعنية للحل."},
            {"name": "الاستدامة", "description": "مدى قابلية الحل للاستمرار والعمل بفعالية على المدى الطويل."}
        ]
        for criterion_data in default_criteria:
            criterion = SolutionEvaluationCriterion(name=criterion_data["name"],
                                                    description=criterion_data["description"])
            db.session.add(criterion)
        try:
            db.session.commit()
            print("INFO: Default evaluation criteria added.")
        except Exception as e:
            db.session.rollback()
            print(f"ERROR: Error adding default criteria: {e}")


@app.route('/problem/<int:problem_id>')
def problem_details(problem_id):  # ...
    problem = Problem.query.get_or_404(problem_id)
    return render_template('problem_details.html', problem=problem, ProposedSolution=ProposedSolution,
                           ChosenSolution=ChosenSolution, PotentialRootCause=PotentialRootCause,
                           ImplementationPlan=ImplementationPlan, SolutionKPI=SolutionKPI,
                           KPIMeasurement=KPIMeasurement, LessonLearned=LessonLearned)


@app.route('/edit_problem/<int:problem_id>', methods=['GET', 'POST'])
def edit_problem(problem_id):  # ...
    problem_to_edit = Problem.query.get_or_404(problem_id)
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        problem_to_edit.title = request.form['title']
        old_description_initial = problem_to_edit.description_initial
        new_description_initial = request.form.get('description_initial')
        problem_to_edit.domain = request.form.get('domain')
        problem_to_edit.complexity_level = request.form.get('complexity_level')
        problem_to_edit.status = request.form.get('status')
        problem_to_edit.stakeholders_involved = request.form.get('stakeholders_involved')
        problem_to_edit.initial_impact_assessment = request.form.get('initial_impact_assessment')
        problem_to_edit.problem_source = request.form.get('problem_source')
        problem_to_edit.refined_problem_statement_final = request.form.get('refined_problem_statement_final')
        problem_tags_input = request.form.get('problem_tags',
                                              problem_to_edit.problem_tags if problem_to_edit.problem_tags is not None else '')
        date_closed_str = request.form.get('date_closed')
        if date_closed_str:
            problem_to_edit.date_closed = parse_date_string(date_closed_str, "تاريخ الإغلاق", '%Y-%m-%dT%H:%M')
        elif 'date_closed' in request.form and not date_closed_str:
            problem_to_edit.date_closed = None
        if new_description_initial != old_description_initial:
            problem_to_edit.description_initial = new_description_initial
            score, label = calculate_sentiment(new_description_initial)
            problem_to_edit.sentiment_score = score
            problem_to_edit.sentiment_label = label
            if new_description_initial:
                lang_for_summary = "english"
                if any('\u0600' <= char <= '\u06FF' for char in new_description_initial[:100]):
                    lang_for_summary = "arabic"
                problem_to_edit.ai_generated_summary = generate_ai_summary(new_description_initial,
                                                                           language=lang_for_summary, sentences_count=2)
                if not problem_tags_input.strip():
                    suggested_keywords_list = extract_keywords(new_description_initial)
                    problem_to_edit.problem_tags = ", ".join(suggested_keywords_list)
                    if not form_data_to_pass.get('problem_tags'):
                        form_data_to_pass['suggested_tags_on_error'] = suggested_keywords_list
                else:
                    problem_to_edit.problem_tags = problem_tags_input
            else:
                problem_to_edit.problem_tags = ""
                problem_to_edit.ai_generated_summary = None
        else:
            problem_to_edit.problem_tags = problem_tags_input
            if not problem_to_edit.ai_generated_summary and problem_to_edit.description_initial:
                lang_for_summary = "english"
                if any('\u0600' <= char <= '\u06FF' for char in problem_to_edit.description_initial[:100]):
                    lang_for_summary = "arabic"
                problem_to_edit.ai_generated_summary = generate_ai_summary(problem_to_edit.description_initial,
                                                                           language=lang_for_summary, sentences_count=2)
        try:
            db.session.commit()
            flash(f"تم تعديل المشكلة '{problem_to_edit.title}' بنجاح!", 'success')
            return redirect(url_for('problem_details', problem_id=problem_to_edit.id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل المشكلة: {str(e)}", 'danger')
            if new_description_initial and not form_data_to_pass.get('problem_tags') and not problem_tags_input.strip():
                form_data_to_pass['suggested_tags_on_error'] = extract_keywords(new_description_initial)
            return render_template('edit_problem.html', problem=problem_to_edit, form_data=form_data_to_pass)
    else:  # GET
        form_data_to_pass['title'] = problem_to_edit.title
        form_data_to_pass['description_initial'] = problem_to_edit.description_initial or ''
        form_data_to_pass['domain'] = problem_to_edit.domain or ''
        form_data_to_pass['complexity_level'] = problem_to_edit.complexity_level
        form_data_to_pass['status'] = problem_to_edit.status
        form_data_to_pass['stakeholders_involved'] = problem_to_edit.stakeholders_involved or ''
        form_data_to_pass['initial_impact_assessment'] = problem_to_edit.initial_impact_assessment or ''
        form_data_to_pass['problem_source'] = problem_to_edit.problem_source or ''
        form_data_to_pass['refined_problem_statement_final'] = problem_to_edit.refined_problem_statement_final or ''
        if problem_to_edit.date_closed:
            form_data_to_pass['date_closed'] = problem_to_edit.date_closed.strftime('%Y-%m-%dT%H:%M')
        else:
            form_data_to_pass['date_closed'] = ''
        current_tags = problem_to_edit.problem_tags
        if not current_tags and problem_to_edit.description_initial:
            suggested_keywords_list = extract_keywords(problem_to_edit.description_initial)
            form_data_to_pass['problem_tags'] = ", ".join(suggested_keywords_list)
            form_data_to_pass['suggested_initial_tags_list'] = suggested_keywords_list
        else:
            form_data_to_pass['problem_tags'] = current_tags if current_tags is not None else ''
        return render_template('edit_problem.html', problem=problem_to_edit, form_data=form_data_to_pass)


@app.route('/problem/<int:problem_id>/understanding', methods=['GET', 'POST'])
def manage_understanding(problem_id):
    problem = Problem.query.get_or_404(problem_id)
    understanding_data = problem.understanding_details
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        if not understanding_data:
            understanding_data = ProblemUnderstanding(problem_id=problem_id)
            db.session.add(understanding_data)
        understanding_data.active_listening_notes = request.form.get('active_listening_notes')
        understanding_data.key_questions_asked = request.form.get('key_questions_asked')
        understanding_data.initial_data_sources = request.form.get('initial_data_sources')
        understanding_data.initial_hypotheses = request.form.get('initial_hypotheses')
        understanding_data.stakeholder_feedback_initial = request.form.get('stakeholder_feedback_initial')
        understanding_data.refined_problem_statement_early = request.form.get('refined_problem_statement_early')
        try:
            db.session.commit()
            flash("تم حفظ/تحديث تفاصيل فهم المشكلة بنجاح!", 'success')
            return redirect(url_for('problem_details', problem_id=problem_id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ: {str(e)}", 'danger')
            return render_template('manage_understanding.html', problem=problem, understanding=understanding_data,
                                   form_data=form_data_to_pass)
    else:
        if understanding_data:
            form_data_to_pass['active_listening_notes'] = understanding_data.active_listening_notes or ''
            form_data_to_pass['key_questions_asked'] = understanding_data.key_questions_asked or ''
            form_data_to_pass['initial_data_sources'] = understanding_data.initial_data_sources or ''
            form_data_to_pass['initial_hypotheses'] = understanding_data.initial_hypotheses or ''
            form_data_to_pass['stakeholder_feedback_initial'] = understanding_data.stakeholder_feedback_initial or ''
            form_data_to_pass[
                'refined_problem_statement_early'] = understanding_data.refined_problem_statement_early or ''
    return render_template('manage_understanding.html', problem=problem, understanding=understanding_data,
                           form_data=form_data_to_pass)


@app.route('/problem/<int:problem_id>/cause_analysis', methods=['GET', 'POST'])
def manage_cause_analysis(problem_id):  # <<< تم تعديل هذا المسار ليشمل اقتراح الأسباب الجذرية >>>
    problem = Problem.query.get_or_404(problem_id)
    analysis_data = problem.cause_analysis_details
    form_data_to_pass = {}
    suggested_rc = []

    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        if not analysis_data:
            analysis_data = CauseAnalysis(problem_id=problem_id)
            db.session.add(analysis_data)
        analysis_data.data_collection_methods_deep = request.form.get('data_collection_methods_deep')
        analysis_data.data_analysis_techniques_used = request.form.get('data_analysis_techniques_used')
        analysis_data.key_findings_from_analysis = request.form.get('key_findings_from_analysis')
        problem.refined_problem_statement_final = request.form.get('refined_problem_statement_final_main')
        try:
            db.session.commit()
            flash("تم حفظ/تحديث تفاصيل تحليل الأسباب بنجاح!", 'success')
            # بعد الحفظ، نعيد حساب الاقتراحات لعرضها مع البيانات المحدثة
            suggested_rc = get_suggested_root_causes(problem.description_initial, problem.problem_tags)
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء حفظ تحليل الأسباب: {str(e)}", 'danger')
            suggested_rc = get_suggested_root_causes(problem.description_initial, problem.problem_tags)
            return render_template('manage_cause_analysis.html',
                                   problem=problem,
                                   analysis=analysis_data,
                                   form_data=form_data_to_pass,
                                   PotentialRootCause=PotentialRootCause,
                                   suggested_root_causes=suggested_rc)
    else:  # GET request
        if analysis_data:
            form_data_to_pass['data_collection_methods_deep'] = analysis_data.data_collection_methods_deep or ''
            form_data_to_pass['data_analysis_techniques_used'] = analysis_data.data_analysis_techniques_used or ''
            form_data_to_pass['key_findings_from_analysis'] = analysis_data.key_findings_from_analysis or ''
        form_data_to_pass['refined_problem_statement_final_main'] = problem.refined_problem_statement_final or ''

        suggested_rc = get_suggested_root_causes(problem.description_initial, problem.problem_tags)

    return render_template('manage_cause_analysis.html',
                           problem=problem,
                           analysis=analysis_data,
                           PotentialRootCause=PotentialRootCause,
                           form_data=form_data_to_pass,
                           suggested_root_causes=suggested_rc)


@app.route('/problem/<int:problem_id>/cause_analysis/add_root_cause', methods=['POST'])
def add_root_cause(problem_id):
    problem = Problem.query.get_or_404(problem_id)
    analysis_data = problem.cause_analysis_details
    if not analysis_data:
        flash("يجب حفظ تفاصيل التحليل الأساسية أولاً لإضافة أسباب جذرية.", 'warning')
        return redirect(url_for('manage_cause_analysis', problem_id=problem_id))
    if request.method == 'POST':
        cause_description = request.form.get('cause_description')
        evidence_supporting_cause = request.form.get('evidence_supporting_cause')
        validation_status = request.form.get('validation_status', 'محتمل')
        impact_of_cause = request.form.get('impact_of_cause')
        if not cause_description:
            flash("وصف السبب الجذري مطلوب.", 'danger')
        else:
            new_root_cause = PotentialRootCause(analysis_id=analysis_data.id, cause_description=cause_description,
                                                evidence_supporting_cause=evidence_supporting_cause,
                                                validation_status=validation_status, impact_of_cause=impact_of_cause)
            try:
                db.session.add(new_root_cause)
                db.session.commit()
                flash("تمت إضافة سبب جذري جديد بنجاح!", 'success')
            except Exception as e:
                db.session.rollback()
                flash(f"حدث خطأ أثناء إضافة السبب الجذري: {str(e)}", 'danger')
        return redirect(url_for('manage_cause_analysis', problem_id=problem_id))


@app.route('/problem/<int:problem_id>/cause_analysis/root_cause/<int:cause_id>/edit', methods=['GET', 'POST'])
def edit_root_cause(problem_id, cause_id):
    problem = Problem.query.get_or_404(problem_id)
    cause_to_edit = PotentialRootCause.query.get_or_404(cause_id)
    if not problem.cause_analysis_details or cause_to_edit.analysis_id != problem.cause_analysis_details.id:
        flash("السبب الجذري غير مرتبط بهذه المشكلة.", 'danger')
        return redirect(url_for('manage_cause_analysis', problem_id=problem_id))
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        cause_to_edit.cause_description = request.form.get('cause_description', cause_to_edit.cause_description)
        cause_to_edit.evidence_supporting_cause = request.form.get('evidence_supporting_cause',
                                                                   cause_to_edit.evidence_supporting_cause)
        cause_to_edit.validation_status = request.form.get('validation_status', cause_to_edit.validation_status)
        cause_to_edit.impact_of_cause = request.form.get('impact_of_cause', cause_to_edit.impact_of_cause)
        if not cause_to_edit.cause_description:
            flash("وصف السبب الجذري مطلوب.", 'danger')
            return render_template('edit_root_cause.html', problem=problem, cause=cause_to_edit,
                                   form_data=form_data_to_pass)
        try:
            db.session.commit()
            flash("تم تعديل السبب الجذري بنجاح!", 'success')
            return redirect(url_for('manage_cause_analysis', problem_id=problem_id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل السبب الجذري: {str(e)}", 'danger')
            return render_template('edit_root_cause.html', problem=problem, cause=cause_to_edit,
                                   form_data=form_data_to_pass)
    else:
        form_data_to_pass['cause_description'] = cause_to_edit.cause_description
        form_data_to_pass['evidence_supporting_cause'] = cause_to_edit.evidence_supporting_cause or ''
        form_data_to_pass['validation_status'] = cause_to_edit.validation_status
        form_data_to_pass['impact_of_cause'] = cause_to_edit.impact_of_cause or ''
    return render_template('edit_root_cause.html', problem=problem, cause=cause_to_edit, form_data=form_data_to_pass)


@app.route('/problem/<int:problem_id>/cause_analysis/delete_root_cause/<int:cause_id>', methods=['POST'])
def delete_root_cause(problem_id, cause_id):
    problem = Problem.query.get_or_404(problem_id)
    if not problem.cause_analysis_details:
        flash("لا يوجد تحليل أسباب لهذه المشكلة.", 'warning')
        return redirect(url_for('problem_details', problem_id=problem_id))
    cause_to_delete = PotentialRootCause.query.get_or_404(cause_id)
    if cause_to_delete.analysis_id != problem.cause_analysis_details.id:
        flash("محاولة حذف سبب جذري غير مصرح بها.", 'danger')
        return redirect(url_for('manage_cause_analysis', problem_id=problem_id))
    try:
        db.session.delete(cause_to_delete)
        db.session.commit()
        flash("تم حذف السبب الجذري بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء حذف السبب الجذري: {str(e)}", 'danger')
    return redirect(url_for('manage_cause_analysis', problem_id=problem_id))


@app.route('/problem/<int:problem_id>/solutions', methods=['GET', 'POST'])
def manage_solutions(problem_id):
    problem = Problem.query.get_or_404(problem_id)
    root_causes = []
    if problem.cause_analysis_details:
        root_causes = problem.cause_analysis_details.potential_root_causes.filter(
            PotentialRootCause.validation_status.in_(['مؤكد', 'محتمل'])).all()
    if request.method == 'POST':
        solution_description = request.form.get('solution_description')
        generation_method = request.form.get('generation_method')
        estimated_cost = request.form.get('estimated_cost')
        estimated_time_to_implement = request.form.get('estimated_time_to_implement')
        potential_benefits = request.form.get('potential_benefits')
        potential_risks = request.form.get('potential_risks')
        if not solution_description:
            flash("وصف الحل المقترح مطلوب.", 'danger')
        else:
            new_solution = ProposedSolution(problem_id=problem.id, solution_description=solution_description,
                                            generation_method=generation_method, estimated_cost=estimated_cost,
                                            estimated_time_to_implement=estimated_time_to_implement,
                                            potential_benefits=potential_benefits, potential_risks=potential_risks)
            try:
                db.session.add(new_solution)
                db.session.commit()
                flash("تمت إضافة حل مقترح جديد بنجاح!", 'success')
            except Exception as e:
                db.session.rollback()
                flash(f"حدث خطأ أثناء إضافة الحل المقترح: {str(e)}", 'danger')
        return redirect(url_for('manage_solutions', problem_id=problem_id))
    proposed_solutions_list = problem.proposed_solutions.order_by(ProposedSolution.id).all()
    return render_template('manage_solutions.html', problem=problem, proposed_solutions=proposed_solutions_list,
                           root_causes=root_causes, ChosenSolution=ChosenSolution, ProposedSolution=ProposedSolution)


@app.route('/problem/<int:problem_id>/solution/<int:solution_id>/edit', methods=['GET', 'POST'])
def edit_proposed_solution(problem_id, solution_id):
    problem = Problem.query.get_or_404(problem_id)
    solution_to_edit = ProposedSolution.query.get_or_404(solution_id)
    if solution_to_edit.problem_id != problem.id:
        flash("الحل المقترح غير مرتبط بهذه المشكلة.", 'danger')
        return redirect(url_for('manage_solutions', problem_id=problem_id))
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        solution_to_edit.solution_description = request.form.get('solution_description',
                                                                 solution_to_edit.solution_description)
        solution_to_edit.generation_method = request.form.get('generation_method', solution_to_edit.generation_method)
        solution_to_edit.estimated_cost = request.form.get('estimated_cost', solution_to_edit.estimated_cost)
        solution_to_edit.estimated_time_to_implement = request.form.get('estimated_time_to_implement',
                                                                        solution_to_edit.estimated_time_to_implement)
        solution_to_edit.potential_benefits = request.form.get('potential_benefits',
                                                               solution_to_edit.potential_benefits)
        solution_to_edit.potential_risks = request.form.get('potential_risks', solution_to_edit.potential_risks)
        if not solution_to_edit.solution_description:
            flash("وصف الحل المقترح مطلوب.", 'danger')
            return render_template('edit_proposed_solution.html', problem=problem, solution=solution_to_edit,
                                   form_data=form_data_to_pass)
        try:
            db.session.commit()
            flash("تم تعديل الحل المقترح بنجاح!", 'success')
            return redirect(url_for('manage_solutions', problem_id=problem_id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل الحل المقترح: {str(e)}", 'danger')
            return render_template('edit_proposed_solution.html', problem=problem, solution=solution_to_edit,
                                   form_data=form_data_to_pass)
    else:  # GET
        form_data_to_pass['solution_description'] = solution_to_edit.solution_description
        form_data_to_pass['generation_method'] = solution_to_edit.generation_method or ''
        form_data_to_pass['estimated_cost'] = solution_to_edit.estimated_cost or ''
        form_data_to_pass['estimated_time_to_implement'] = solution_to_edit.estimated_time_to_implement or ''
        form_data_to_pass['potential_benefits'] = solution_to_edit.potential_benefits or ''
        form_data_to_pass['potential_risks'] = solution_to_edit.potential_risks or ''
    return render_template('edit_proposed_solution.html', problem=problem, solution=solution_to_edit,
                           form_data=form_data_to_pass)


@app.route('/problem/<int:problem_id>/solution/<int:solution_id>/delete', methods=['POST'])
def delete_proposed_solution(problem_id, solution_id):
    problem = Problem.query.get_or_404(problem_id)
    solution_to_delete = ProposedSolution.query.get_or_404(solution_id)
    if solution_to_delete.problem_id != problem.id:
        flash("محاولة حذف حل غير مصرح بها.", 'danger')
        return redirect(url_for('manage_solutions', problem_id=problem_id))
    try:
        db.session.delete(solution_to_delete)
        db.session.commit()
        flash("تم حذف الحل المقترح بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء حذف الحل المقترح: {str(e)}", 'danger')
    return redirect(url_for('manage_solutions', problem_id=problem_id))


@app.route('/problem/<int:problem_id>/solution/<int:solution_id>/choose', methods=['POST'])
def choose_solution(problem_id, solution_id):
    problem = Problem.query.get_or_404(problem_id)
    solution_to_choose = ProposedSolution.query.get_or_404(solution_id)
    if solution_to_choose.problem_id != problem.id:
        flash("محاولة اختيار حل غير مصرح بها.", 'danger')
        return redirect(url_for('manage_solutions', problem_id=problem_id))
    existing_chosen_record = ChosenSolution.query.filter_by(problem_id=problem.id,
                                                            proposed_solution_id=solution_to_choose.id).first()
    if existing_chosen_record:
        flash("هذا الحل تم اختياره بالفعل.", "warning")
        return redirect(url_for('manage_solutions', problem_id=problem_id))
    justification = request.form.get('justification_for_choice', "تم اختياره بناءً على التقييم.")
    new_chosen_solution = ChosenSolution(problem_id=problem.id, proposed_solution_id=solution_to_choose.id,
                                         justification_for_choice=justification, approval_status="مختار")
    solution_to_choose.is_chosen = True
    try:
        db.session.add(new_chosen_solution)
        db.session.commit()
        flash(f"تم اختيار الحل '{solution_to_choose.solution_description[:30]}...' بنجاح!", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء اختيار الحل: {str(e)}", 'danger')
    return redirect(url_for('manage_solutions', problem_id=problem_id))


@app.route('/chosen_solution/<int:chosen_solution_id>/edit_details', methods=['GET', 'POST'])
def edit_chosen_solution_details(chosen_solution_id):
    chosen_solution_to_edit = ChosenSolution.query.get_or_404(chosen_solution_id)
    problem = chosen_solution_to_edit.problem_parent
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        chosen_solution_to_edit.justification_for_choice = request.form.get('justification_for_choice',
                                                                            chosen_solution_to_edit.justification_for_choice)
        chosen_solution_to_edit.approval_status = request.form.get('approval_status',
                                                                   chosen_solution_to_edit.approval_status)
        try:
            db.session.commit()
            flash("تم تعديل تفاصيل الحل المختار بنجاح!", 'success')
            return redirect(url_for('manage_solutions', problem_id=problem.id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل تفاصيل الحل المختار: {str(e)}", 'danger')
            return render_template('edit_chosen_solution_details.html', chosen_solution=chosen_solution_to_edit,
                                   problem=problem, form_data=form_data_to_pass)
    else:  # GET
        form_data_to_pass['justification_for_choice'] = chosen_solution_to_edit.justification_for_choice or ''
        form_data_to_pass['approval_status'] = chosen_solution_to_edit.approval_status or ''
        if chosen_solution_to_edit.date_chosen:
            form_data_to_pass['date_chosen'] = chosen_solution_to_edit.date_chosen.strftime('%Y-%m-%dT%H:%M')
        else:
            form_data_to_pass['date_chosen'] = ''
    return render_template('edit_chosen_solution_details.html', chosen_solution=chosen_solution_to_edit,
                           problem=problem, form_data=form_data_to_pass)


@app.route('/problem/<int:problem_id>/chosen_solution/<int:chosen_solution_id>/unchoose', methods=['POST'])
def unchoose_solution(problem_id, chosen_solution_id):
    problem = Problem.query.get_or_404(problem_id)
    chosen_solution_to_remove = ChosenSolution.query.get_or_404(chosen_solution_id)
    if chosen_solution_to_remove.problem_id != problem.id:
        flash("محاولة إلغاء اختيار حل غير مصرح بها.", 'danger')
        return redirect(url_for('manage_solutions', problem_id=problem_id))
    if chosen_solution_to_remove.proposed_solution_details:
        chosen_solution_to_remove.proposed_solution_details.is_chosen = False
    try:
        db.session.delete(chosen_solution_to_remove)
        db.session.commit()
        flash("تم إلغاء اختيار الحل بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء إلغاء اختيار الحل: {str(e)}", 'danger')
    return redirect(url_for('manage_solutions', problem_id=problem_id))


@app.route('/chosen_solution/<int:chosen_solution_id>/implementation_plan', methods=['GET', 'POST'])
def manage_implementation_plan(chosen_solution_id):
    chosen_solution = ChosenSolution.query.get_or_404(chosen_solution_id)
    problem = chosen_solution.problem_parent
    implementation_plan = chosen_solution.implementation_plan
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        if not implementation_plan:
            implementation_plan = ImplementationPlan(chosen_solution_id=chosen_solution.id)
            db.session.add(implementation_plan)
        implementation_plan.plan_description = request.form.get('plan_description')
        implementation_plan.overall_status = request.form.get('overall_status', 'لم يبدأ')
        implementation_plan.overall_budget = request.form.get('overall_budget')
        implementation_plan.key_personnel = request.form.get('key_personnel')
        implementation_plan.start_date_planned = parse_date_string(request.form.get('start_date_planned'),
                                                                   "تاريخ البدء المخطط")
        implementation_plan.end_date_planned = parse_date_string(request.form.get('end_date_planned'),
                                                                 "تاريخ الانتهاء المخطط")
        implementation_plan.start_date_actual = parse_date_string(request.form.get('start_date_actual'),
                                                                  "تاريخ البدء الفعلي")
        implementation_plan.end_date_actual = parse_date_string(request.form.get('end_date_actual'),
                                                                "تاريخ الانتهاء الفعلي")
        try:
            db.session.commit()
            flash("تم حفظ/تحديث خطة التنفيذ بنجاح!", 'success')
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء حفظ خطة التنفيذ: {str(e)}", 'danger')
    else:
        if implementation_plan:
            form_data_to_pass['plan_description'] = implementation_plan.plan_description or ''
            form_data_to_pass['overall_status'] = implementation_plan.overall_status
            form_data_to_pass['overall_budget'] = implementation_plan.overall_budget or ''
            form_data_to_pass['key_personnel'] = implementation_plan.key_personnel or ''
            if implementation_plan.start_date_planned: form_data_to_pass[
                'start_date_planned'] = implementation_plan.start_date_planned.strftime('%Y-%m-%d')
            if implementation_plan.end_date_planned: form_data_to_pass[
                'end_date_planned'] = implementation_plan.end_date_planned.strftime('%Y-%m-%d')
            if implementation_plan.start_date_actual: form_data_to_pass[
                'start_date_actual'] = implementation_plan.start_date_actual.strftime('%Y-%m-%d')
            if implementation_plan.end_date_actual: form_data_to_pass[
                'end_date_actual'] = implementation_plan.end_date_actual.strftime('%Y-%m-%d')
    tasks = []
    if implementation_plan and implementation_plan.id:
        tasks = implementation_plan.tasks.order_by(ImplementationTask.due_date.asc().nulls_last(),
                                                   ImplementationTask.id.asc()).all()
    return render_template('manage_implementation_plan.html', problem=problem, chosen_solution=chosen_solution,
                           plan=implementation_plan, tasks=tasks, form_data=form_data_to_pass)


@app.route('/implementation_plan/<int:plan_id>/add_task', methods=['POST'])
def add_implementation_task(plan_id):
    plan = ImplementationPlan.query.get_or_404(plan_id)
    chosen_solution = plan.chosen_solution_parent
    if request.method == 'POST':
        task_description = request.form.get('task_description')
        assigned_to = request.form.get('assigned_to')
        priority = request.form.get('priority', 'متوسط')
        task_status = request.form.get('task_status', 'لم تبدأ')
        notes = request.form.get('notes')
        due_date_obj = parse_date_string(request.form.get('due_date'), "تاريخ استحقاق المهمة")
        if not task_description:
            flash("وصف المهمة مطلوب.", 'danger')
        else:
            new_task = ImplementationTask(plan_id=plan.id, task_description=task_description, assigned_to=assigned_to,
                                          priority=priority, task_status=task_status, due_date=due_date_obj,
                                          notes=notes)
            try:
                db.session.add(new_task)
                db.session.commit()
                flash("تمت إضافة مهمة تنفيذ جديدة بنجاح!", 'success')
            except Exception as e:
                db.session.rollback()
                flash(f"حدث خطأ أثناء إضافة المهمة: {str(e)}", 'danger')
        return redirect(url_for('manage_implementation_plan', chosen_solution_id=chosen_solution.id))


@app.route('/implementation_task/<int:task_id>/edit', methods=['GET', 'POST'])
def edit_implementation_task(task_id):
    task_to_edit = ImplementationTask.query.get_or_404(task_id)
    plan = task_to_edit.plan_parent
    chosen_solution = plan.chosen_solution_parent
    problem = chosen_solution.problem_parent
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        task_to_edit.task_description = request.form.get('task_description', task_to_edit.task_description)
        task_to_edit.assigned_to = request.form.get('assigned_to', task_to_edit.assigned_to)
        task_to_edit.priority = request.form.get('priority', task_to_edit.priority)
        task_to_edit.due_date = parse_date_string(request.form.get('due_date'), "تاريخ الاستحقاق")
        task_to_edit.notes = request.form.get('notes', task_to_edit.notes)
        if not task_to_edit.task_description:
            flash("وصف المهمة مطلوب.", 'danger')
            return render_template('edit_implementation_task.html', task=task_to_edit, problem=problem,
                                   chosen_solution=chosen_solution, plan=plan, form_data=form_data_to_pass)
        try:
            db.session.commit()
            flash("تم تعديل مهمة التنفيذ بنجاح!", 'success')
            return redirect(url_for('manage_implementation_plan', chosen_solution_id=chosen_solution.id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل المهمة: {str(e)}", 'danger')
            return render_template('edit_implementation_task.html', task=task_to_edit, problem=problem,
                                   chosen_solution=chosen_solution, plan=plan, form_data=form_data_to_pass)
    else:  # GET
        form_data_to_pass['task_description'] = task_to_edit.task_description
        form_data_to_pass['assigned_to'] = task_to_edit.assigned_to or ''
        form_data_to_pass['priority'] = task_to_edit.priority
        if task_to_edit.due_date:
            form_data_to_pass['due_date'] = task_to_edit.due_date.strftime('%Y-%m-%d')
        else:
            form_data_to_pass['due_date'] = ''
        form_data_to_pass['notes'] = task_to_edit.notes or ''
    return render_template('edit_implementation_task.html', task=task_to_edit, problem=problem,
                           chosen_solution=chosen_solution, plan=plan, form_data=form_data_to_pass)


@app.route('/implementation_task/<int:task_id>/update_status', methods=['POST'])
def update_task_status(task_id):
    task = ImplementationTask.query.get_or_404(task_id)
    new_status = request.form.get('task_status_update')
    chosen_solution_id = task.plan_parent.chosen_solution_id
    if new_status:
        task.task_status = new_status
        if new_status == 'مكتملة' and not task.completion_date:
            task.completion_date = datetime.datetime.utcnow()
        elif new_status != 'مكتملة':
            task.completion_date = None
        try:
            db.session.commit()
            flash(f"تم تحديث حالة المهمة '{task.task_description[:20]}...' إلى '{new_status}'.", 'success')
        except Exception as e:
            db.session.rollback()
            flash(f"خطأ في تحديث حالة المهمة: {str(e)}", 'danger')
    else:
        flash("لم يتم توفير حالة جديدة للمهمة.", 'warning')
    return redirect(url_for('manage_implementation_plan', chosen_solution_id=chosen_solution_id))


@app.route('/implementation_task/<int:task_id>/delete', methods=['POST'])
def delete_implementation_task(task_id):
    task_to_delete = ImplementationTask.query.get_or_404(task_id)
    chosen_solution_id = task_to_delete.plan_parent.chosen_solution_id
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        flash("تم حذف مهمة التنفيذ بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء حذف المهمة: {str(e)}", 'danger')
    return redirect(url_for('manage_implementation_plan', chosen_solution_id=chosen_solution_id))


@app.route('/chosen_solution/<int:chosen_solution_id>/kpis', methods=['GET', 'POST'])
def manage_kpis(chosen_solution_id):
    chosen_solution = ChosenSolution.query.get_or_404(chosen_solution_id)
    problem = chosen_solution.problem_parent
    if request.method == 'POST':
        kpi_name = request.form.get('kpi_name')
        kpi_description = request.form.get('kpi_description')
        target_value = request.form.get('target_value')
        current_value_baseline = request.form.get('current_value_baseline')
        measurement_unit = request.form.get('measurement_unit')
        measurement_frequency = request.form.get('measurement_frequency')
        if not kpi_name:
            flash("اسم مؤشر الأداء مطلوب.", 'danger')
        else:
            new_kpi = SolutionKPI(chosen_solution_id=chosen_solution.id, kpi_name=kpi_name,
                                  kpi_description=kpi_description, target_value=target_value,
                                  current_value_baseline=current_value_baseline, measurement_unit=measurement_unit,
                                  measurement_frequency=measurement_frequency)
            try:
                db.session.add(new_kpi)
                db.session.commit()
                flash(f"تمت إضافة مؤشر أداء جديد: '{kpi_name}' بنجاح!", 'success')
            except Exception as e:
                db.session.rollback()
                flash(f"حدث خطأ أثناء إضافة مؤشر الأداء: {str(e)}", 'danger')
        return redirect(url_for('manage_kpis', chosen_solution_id=chosen_solution_id))
    kpis_list = chosen_solution.kpis.order_by(SolutionKPI.id).all()
    return render_template('manage_kpis.html', problem=problem, chosen_solution=chosen_solution, kpis=kpis_list,
                           datetime=datetime, KPIMeasurement=KPIMeasurement)


@app.route('/kpi/<int:kpi_id>/edit', methods=['GET', 'POST'])
def edit_kpi(kpi_id):
    kpi_to_edit = SolutionKPI.query.get_or_404(kpi_id)
    chosen_solution = kpi_to_edit.chosen_solution_parent
    problem = chosen_solution.problem_parent
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        kpi_to_edit.kpi_name = request.form.get('kpi_name', kpi_to_edit.kpi_name)
        kpi_to_edit.kpi_description = request.form.get('kpi_description', kpi_to_edit.kpi_description)
        kpi_to_edit.target_value = request.form.get('target_value', kpi_to_edit.target_value)
        kpi_to_edit.current_value_baseline = request.form.get('current_value_baseline',
                                                              kpi_to_edit.current_value_baseline)
        kpi_to_edit.measurement_unit = request.form.get('measurement_unit', kpi_to_edit.measurement_unit)
        kpi_to_edit.measurement_frequency = request.form.get('measurement_frequency', kpi_to_edit.measurement_frequency)
        if not kpi_to_edit.kpi_name:
            flash("اسم مؤشر الأداء مطلوب.", 'danger')
            return render_template('edit_kpi.html', kpi=kpi_to_edit, chosen_solution=chosen_solution, problem=problem,
                                   form_data=form_data_to_pass)
        try:
            db.session.commit()
            flash(f"تم تعديل مؤشر الأداء '{kpi_to_edit.kpi_name}' بنجاح!", 'success')
            return redirect(url_for('manage_kpis', chosen_solution_id=chosen_solution.id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل مؤشر الأداء: {str(e)}", 'danger')
            return render_template('edit_kpi.html', kpi=kpi_to_edit, chosen_solution=chosen_solution, problem=problem,
                                   form_data=form_data_to_pass)
    else:  # GET
        form_data_to_pass['kpi_name'] = kpi_to_edit.kpi_name
        form_data_to_pass['kpi_description'] = kpi_to_edit.kpi_description or ''
        form_data_to_pass['target_value'] = kpi_to_edit.target_value or ''
        form_data_to_pass['current_value_baseline'] = kpi_to_edit.current_value_baseline or ''
        form_data_to_pass['measurement_unit'] = kpi_to_edit.measurement_unit or ''
        form_data_to_pass['measurement_frequency'] = kpi_to_edit.measurement_frequency or ''
    return render_template('edit_kpi.html', kpi=kpi_to_edit, chosen_solution=chosen_solution, problem=problem,
                           form_data=form_data_to_pass)


@app.route('/kpi/<int:kpi_id>/add_measurement', methods=['POST'])
def add_kpi_measurement(kpi_id):
    kpi = SolutionKPI.query.get_or_404(kpi_id)
    chosen_solution_id = kpi.chosen_solution_id
    if request.method == 'POST':
        actual_value = request.form.get('actual_value')
        notes = request.form.get('notes')
        measurement_date_str = request.form.get('measurement_date')
        measurement_date_obj = parse_date_string(measurement_date_str, "تاريخ القياس")
        if measurement_date_obj is None and not measurement_date_str:
            measurement_date_obj = datetime.datetime.utcnow()
        if not actual_value:
            flash("القيمة الفعلية للقياس مطلوبة.", 'danger')
        elif measurement_date_obj is None and measurement_date_str:
            pass
        else:
            new_measurement = KPIMeasurement(kpi_id=kpi.id, actual_value=actual_value,
                                             measurement_date=measurement_date_obj, notes=notes)
            try:
                db.session.add(new_measurement)
                db.session.commit()
                flash("تمت إضافة قياس جديد للمؤشر بنجاح!", 'success')
            except Exception as e:
                db.session.rollback()
                flash(f"حدث خطأ أثناء إضافة القياس: {str(e)}", 'danger')
        return redirect(url_for('manage_kpis', chosen_solution_id=chosen_solution_id))


@app.route('/measurement/<int:measurement_id>/edit', methods=['GET', 'POST'])
def edit_kpi_measurement(measurement_id):
    measurement_to_edit = KPIMeasurement.query.get_or_404(measurement_id)
    kpi = measurement_to_edit.kpi_parent
    chosen_solution = kpi.chosen_solution_parent
    problem = chosen_solution.problem_parent
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        measurement_to_edit.actual_value = request.form.get('actual_value', measurement_to_edit.actual_value)
        new_date_obj = parse_date_string(request.form.get('measurement_date'), "تاريخ القياس")
        if new_date_obj:
            measurement_to_edit.measurement_date = new_date_obj
        measurement_to_edit.notes = request.form.get('notes', measurement_to_edit.notes)
        if not measurement_to_edit.actual_value:
            flash("القيمة الفعلية للقياس مطلوبة.", 'danger')
            return render_template('edit_kpi_measurement.html', measurement=measurement_to_edit, kpi=kpi,
                                   chosen_solution=chosen_solution, problem=problem, form_data=form_data_to_pass)
        try:
            db.session.commit()
            flash("تم تعديل قياس المؤشر بنجاح!", 'success')
            return redirect(url_for('manage_kpis', chosen_solution_id=chosen_solution.id))
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء تعديل قياس المؤشر: {str(e)}", 'danger')
            return render_template('edit_kpi_measurement.html', measurement=measurement_to_edit, kpi=kpi,
                                   chosen_solution=chosen_solution, problem=problem, form_data=form_data_to_pass)
    else:  # GET
        form_data_to_pass['actual_value'] = measurement_to_edit.actual_value
        if measurement_to_edit.measurement_date:
            form_data_to_pass['measurement_date'] = measurement_to_edit.measurement_date.strftime('%Y-%m-%d')
        else:
            form_data_to_pass['measurement_date'] = ''
        form_data_to_pass['notes'] = measurement_to_edit.notes or ''
    return render_template('edit_kpi_measurement.html', measurement=measurement_to_edit, kpi=kpi,
                           chosen_solution=chosen_solution, problem=problem, form_data=form_data_to_pass)


@app.route('/kpi/<int:kpi_id>/delete', methods=['POST'])
def delete_kpi(kpi_id):
    kpi_to_delete = SolutionKPI.query.get_or_404(kpi_id)
    chosen_solution_id = kpi_to_delete.chosen_solution_id
    try:
        db.session.delete(kpi_to_delete)
        db.session.commit()
        flash("تم حذف مؤشر الأداء وقياساته بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء حذف مؤشر الأداء: {str(e)}", 'danger')
    return redirect(url_for('manage_kpis', chosen_solution_id=chosen_solution_id))


@app.route('/measurement/<int:measurement_id>/delete', methods=['POST'])
def delete_kpi_measurement(measurement_id):
    measurement_to_delete = KPIMeasurement.query.get_or_404(measurement_id)
    chosen_solution_id = measurement_to_delete.kpi_parent.chosen_solution_id
    try:
        db.session.delete(measurement_to_delete)
        db.session.commit()
        flash("تم حذف القياس بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"خطأ أثناء حذف القياس: {str(e)}", 'danger')
    return redirect(url_for('manage_kpis', chosen_solution_id=chosen_solution_id))


@app.route('/problem/<int:problem_id>/lessons_learned', methods=['GET', 'POST'])
def manage_lessons_learned(problem_id):
    problem = Problem.query.get_or_404(problem_id)
    lessons_data = problem.lessons_learned
    form_data_to_pass = {}
    if request.method == 'POST':
        form_data_to_pass = request.form.to_dict()
        if not lessons_data:
            lessons_data = LessonLearned(problem_id=problem.id)
            db.session.add(lessons_data)
        lessons_data.what_went_well = request.form.get('what_went_well')
        lessons_data.what_could_be_improved = request.form.get('what_could_be_improved')
        lessons_data.recommendations_for_future = request.form.get('recommendations_for_future')
        lessons_data.key_takeaways = request.form.get('key_takeaways')
        try:
            db.session.commit()
            flash("تم حفظ/تحديث الدروس المستفادة بنجاح!", 'success')
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء حفظ الدروس المستفادة: {str(e)}", 'danger')
            return render_template('manage_lessons_learned.html', problem=problem, lessons=lessons_data,
                                   form_data=form_data_to_pass)
    else:
        if lessons_data:
            form_data_to_pass['what_went_well'] = lessons_data.what_went_well or ''
            form_data_to_pass['what_could_be_improved'] = lessons_data.what_could_be_improved or ''
            form_data_to_pass['recommendations_for_future'] = lessons_data.recommendations_for_future or ''
            form_data_to_pass['key_takeaways'] = lessons_data.key_takeaways or ''
    return render_template('manage_lessons_learned.html', problem=problem, lessons=lessons_data,
                           form_data=form_data_to_pass)


@app.route('/problem/<int:problem_id>/delete', methods=['POST'])
def delete_problem(problem_id):
    problem_to_delete = Problem.query.get_or_404(problem_id)
    try:
        db.session.delete(problem_to_delete)
        db.session.commit()
        flash(f"تم حذف المشكلة '{problem_to_delete.title}' وجميع بياناتها المرتبطة بنجاح.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"حدث خطأ أثناء محاولة حذف المشكلة: {str(e)}", 'danger')
    return redirect(url_for('index'))


@app.route('/problem/<int:problem_id>/close', methods=['POST'])
def close_problem(problem_id):
    problem = Problem.query.get_or_404(problem_id)
    confirmation = request.form.get('confirm_close')
    if confirmation == 'yes':
        problem.status = 'مغلقة'
        problem.date_closed = datetime.datetime.utcnow()
        try:
            db.session.commit()
            flash(f"تم إغلاق المشكلة '{problem.title}' بنجاح.", 'success')
        except Exception as e:
            db.session.rollback()
            flash(f"حدث خطأ أثناء إغلاق المشكلة: {str(e)}", 'danger')
    else:
        flash("لم يتم تأكيد إغلاق المشكلة.", 'warning')
    return redirect(url_for('problem_details', problem_id=problem_id))


@app.route('/problem/<int:problem_id>/reopen', methods=['POST'])
def reopen_problem(problem_id):
    problem = Problem.query.get_or_404(problem_id)
    problem.status = 'مفتوحة'
    problem.date_closed = None
    try:
        db.session.commit()
        flash(f"تمت إعادة فتح المشكلة '{problem.title}'.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"حدث خطأ أثناء إعادة فتح المشكلة: {str(e)}", 'danger')
    return redirect(url_for('problem_details', problem_id=problem_id))


@app.route('/api/suggest_keywords', methods=['POST'])
def suggest_keywords_api():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing description in request body'}), 400
    description_text = data['description']
    if not description_text.strip():
        return jsonify({'keywords': []})
    keywords = extract_keywords(description_text, max_keywords=7)
    return jsonify({'keywords': keywords})


# <<< بداية: مسار API جديد للـ Chatbot >>>
@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message in request body'}), 400

    user_message = data['message']
    if not user_message.strip():
        return jsonify({'reply': 'الرجاء إدخال سؤال.'})  # رد إذا كانت الرسالة فارغة

    bot_reply = get_chatbot_response(user_message)
    return jsonify({'reply': bot_reply})


# <<< نهاية: مسار API جديد للـ Chatbot >>>

if __name__ == '__main__':
    with app.app_context():
        add_default_criteria_if_empty()
        try:
            import nltk

            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
        try:
            import nltk

            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass

    app.run(host='0.0.0.0', debug=True, port=5010)
