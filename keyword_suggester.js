// static/js/keyword_suggester.js

const ARABIC_STOP_WORDS = [
    "في", "من", "على", "إلى", "عن", "هو", "هي", "كان", "يكون", "قد", "تم", "جدا", "جداً",
    "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "هم", "هن", "أنا", "أنت", "ان", "انه", "انها",
    "أنتم", "أنتن", "نحن", "له", "لها", "لهم", "به", "بها", "بهم", "و", "أو", "ثم", "لكن",
    "إن", "أن", "ما", "لم", "لن", "لا", "يا", "كل", "بعض", "غير", "جعل", "جعله", "جعلها",
    "مع", "عند", "مثل", "حتى", "أي", "لقد", "كما", "بين", "فوق", "تحت", "بعد", "قبل", "اذا",
    "كانت", "كنت", "يكونون", "تكون", "علي", "عليه", "عليها", "عليهم", "الي", "اليك", "اخرى",
    "احد", "اول", "اكثر", "اقل", "اذا", "ال", "الا", "امام", "اما", "خلال", "دون", "نحو",
    "هناك", "هنا", "ذلك", "تلك", "ذات", "ذلكما", "تلكما", "اولئك", "اولالك", "جميع", "جميعا",
    "جدا", "جدا", "حيث", "الان", "اليوم", "امس", "غدا", "متى", "اين", "كيف", "لماذا", "ماذا","الى"

    // قم بتوسيع هذه القائمة بشكل كبير - هذه مجرد بداية
];

function processSuggestedPhrase(phrase) {
    if (!phrase || typeof phrase !== 'string') {
        return [];
    }
    let words = phrase.split(/\s+/);
    let filteredWords = words.map(word => {
        let cleanedWord = word.replace(/^[.,\/#!$%\^&\*;:{}=\-_`~()؟٬٫“۔”«»'"\[\]]+|[.,\/#!$%\^&\*;:{}=\-_`~()؟٬٫“۔”«»'"\[\]]+$/g, "");
        return cleanedWord.toLowerCase();
    }).filter(word => {
        return word.trim().length > 2 && !ARABIC_STOP_WORDS.includes(word.trim());
    });
    return [...new Set(filteredWords)];
}

window.addTagToInput = function(tagPhrase, problemTagsInputElementId) {
    const problemTagsInput = document.getElementById(problemTagsInputElementId);
    if (!problemTagsInput) {
        console.error(`Element with ID '${problemTagsInputElementId}' not found for addTagToInput.`);
        return;
    }

    let currentTagsArray = problemTagsInput.value.split(',')
                                .map(t => t.trim())
                                .filter(t => t.length > 0);

    const processedTags = processSuggestedPhrase(tagPhrase);

    processedTags.forEach(singleTag => {
        if (singleTag && !currentTagsArray.includes(singleTag)) {
            currentTagsArray.push(singleTag);
        }
    });

    problemTagsInput.value = currentTagsArray.join(', ');
};

function initializeKeywordSuggester(descriptionTextareaId, suggestedKeywordsAreaId, problemTagsInputId, suggestKeywordsApiUrl) {
    const descriptionTextarea = document.getElementById(descriptionTextareaId);
    const suggestedKeywordsArea = document.getElementById(suggestedKeywordsAreaId);

    if (!descriptionTextarea || !suggestedKeywordsArea) {
        console.error("Keyword suggester: One or more required HTML elements not found. Check IDs:",
                      descriptionTextareaId, suggestedKeywordsAreaId, problemTagsInputId);
        return;
    }

    let debounceTimer;

    descriptionTextarea.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            const descriptionText = this.value;
            if (descriptionText.trim().length > 15) {
                suggestedKeywordsArea.innerHTML = '<small class="loading-indicator"><i>جاري البحث عن كلمات مفتاحية مقترحة...</i></small>';
                try {
                    const response = await fetch(suggestKeywordsApiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ description: descriptionText })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        suggestedKeywordsArea.innerHTML = '';
                        if (data.keywords && data.keywords.length > 0) {
                            const p = document.createElement('p');
                            p.innerHTML = '<small>كلمات/عبارات مقترحة (انقر للإضافة):</small>';
                            suggestedKeywordsArea.appendChild(p);

                            const ul = document.createElement('ul');
                            ul.className = 'suggested-tags-list';
                            data.keywords.forEach(keywordPhrase => {
                                const li = document.createElement('li');
                                li.textContent = keywordPhrase;
                                li.onclick = function() {
                                    if (typeof window.addTagToInput === 'function') {
                                        window.addTagToInput(keywordPhrase, problemTagsInputId);
                                    } else {
                                        console.error("addTagToInput is not defined globally.");
                                    }
                                };
                                ul.appendChild(li);
                            });
                            suggestedKeywordsArea.appendChild(ul);
                        } else {
                            suggestedKeywordsArea.innerHTML = '<small class="text-muted">لا توجد كلمات مفتاحية مقترحة حاليًا لهذا الوصف.</small>';
                        }
                    } else {
                        console.error('Error from API:', response.status, await response.text());
                        suggestedKeywordsArea.innerHTML = '<small class="text-danger">حدث خطأ أثناء جلب الاقتراحات من الخادم.</small>';
                    }
                } catch (error) {
                    console.error('Error fetching suggested keywords:', error);
                    suggestedKeywordsArea.innerHTML = '<small class="text-danger">حدث خطأ في الاتصال بالخادم لاقتراح الكلمات.</small>';
                }
            } else {
                suggestedKeywordsArea.innerHTML = '';
            }
        }, 800);
    });
}