import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;

enum NgramLanguage { english, french, arabic, kabyle }

class NgramPredictor {
  NgramPredictor(this.language);

  final NgramLanguage language;

  // Cache for loaded probability maps per n (2..5)
  final Map<int, Map<String, dynamic>> _probByN = {};

  // Vocabularies for id <-> word mapping when probability files use token ids
  Map<String, dynamic>? _vocab; // word -> id
  Map<String, dynamic>? _vocabRev; // id -> word
  Map<String, double>? _freqWordProb; // word -> probability
  List<MapEntry<String, double>>? _freqTopSorted; // cached sorted list

  String get _langDir {
    switch (language) {
      case NgramLanguage.english:
        return 'english';
      case NgramLanguage.french:
        return 'french';
      case NgramLanguage.arabic:
        return 'arabic';
      case NgramLanguage.kabyle:
        return 'amazigh';
    }
  }

  /// Check if this language uses log probabilities instead of regular probabilities
  bool get _usesLogProba {
    return language == NgramLanguage.kabyle;
  }

  Future<void> _ensureVocabLoaded() async {
    if (_vocab != null && _vocabRev != null) return;
    final vocabPath = 'assets/ngrams_tp/$_langDir/vocab.json';
    final vocabRevPath = 'assets/ngrams_tp/$_langDir/vocab_rev.json';
    try {
      final raw = await rootBundle.loadString(vocabPath);
      final decoded = json.decode(raw);
      if (decoded is Map<String, dynamic>) {
        _vocab = decoded;
      } else if (decoded is List) {
        final map = <String, dynamic>{};
        for (int i = 0; i < decoded.length; i++) {
          final w = decoded[i]?.toString();
          if (w == null || w.isEmpty) continue;
          map[w] = i.toString();
        }
        _vocab = map;
      } else {
        _vocab = const {};
      }
    } catch (_) {
      _vocab = const {};
    }
    try {
      final raw = await rootBundle.loadString(vocabRevPath);
      final decoded = json.decode(raw);
      if (decoded is Map<String, dynamic>) {
        _vocabRev = decoded;
      } else if (decoded is List) {
        final map = <String, dynamic>{};
        for (int i = 0; i < decoded.length; i++) {
          final w = decoded[i]?.toString();
          if (w == null) continue;
          map[i.toString()] = w;
        }
        _vocabRev = map;
      } else {
        _vocabRev = const {};
      }
    } catch (_) {
      _vocabRev = const {};
    }
  }

  Future<void> _ensureFreqLoaded() async {
    if (_freqWordProb != null) return;
    // Load unigrams from 1gram.json and calculate probabilities
    final path = 'assets/ngrams_tp/$_langDir/1gram.json';
    try {
      final raw = await rootBundle.loadString(path);
      final decoded = json.decode(raw) as Map<String, dynamic>;
      await _ensureVocabLoaded(); // Ensure vocab is loaded first

      // Calculate total count
      double total = 0;
      final counts = <String, double>{};
      decoded.forEach((k, v) {
        final count = _asDouble(v);
        if (count == null) return;
        final word = k.trim();
        if (word.isEmpty) return;
        counts[word] = count;
        total += count;
      });

      // Calculate probabilities
      final out = <String, double>{};
      counts.forEach((word, count) {
        if (total > 0) {
          out[word] = count / total;
        }
      });

      _freqWordProb = out;
      _freqTopSorted = out.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));
    } catch (_) {
      _freqWordProb = const {};
      _freqTopSorted = const [];
    }
  }

  Future<Map<String, dynamic>> _loadProbN(int n) async {
    if (n < 2 || n > 5) {
      throw ArgumentError('Supported n for probability files is 2..5');
    }
    if (_probByN.containsKey(n)) return _probByN[n]!;
    final path = 'assets/ngrams_tp/$_langDir/prob_${n}gram.json';
    final raw = await rootBundle.loadString(path);
    final decoded = json.decode(raw) as Map<String, dynamic>;

    // For log probabilities (Kabyle), keep them as-is and convert per-context when needed
    // This is more efficient and correct (normalization happens per context, not globally)
    _probByN[n] = decoded;
    return decoded;
  }

  /// Returns up to topK next-word suggestions for the given [inputText].
  /// Follows Python example pattern:
  /// - 1 word → use 2-gram (bigram) with that word
  /// - 2+ words → use 3-gram (trigram) with last 2 words
  /// - N words → use (N+1)-gram with last N words (up to 5-gram max)
  Future<List<MapEntry<String, double>>> suggest(
    String inputText, {
    int topK = 3,
  }) async {
    await _ensureVocabLoaded();

    // Tokenize and normalize input (like Python's .lower().split())
    final tokens = _normalizeAndTokenize(inputText, language);

    // Filter out partial/unknown words (only use complete known words)
    List<String> completeWords = [];
    for (int i = 0; i < tokens.length; i++) {
      final token = tokens[i];
      if (_isKnownWord(token)) {
        completeWords.add(token);
      } else {
        // Last token might be partial - skip it
        // If not last, might be rare word, include it anyway
        if (i < tokens.length - 1) {
          completeWords.add(token);
        }
      }
    }

    // If no complete words, use fallback (most frequent unigrams)
    if (completeWords.isEmpty) {
      await _ensureFreqLoaded();
      return _freqTopSorted!.take(topK).toList(growable: false);
    }

    // Match Python pattern:
    // - len == 1: use 2-gram (bigram) with that 1 word
    // - len >= 2: use 3-gram (trigram) with last 2 words
    // - Generalize: use (N+1)-gram with last N words, capped at 5-gram
    final wordCount = completeWords.length;
    final maxN = 5;
    final targetN = (wordCount + 1).clamp(2, maxN);
    final contextSize = targetN - 1;

    // Use last N words as context (like Python's txt[-N:])
    if (wordCount < contextSize) {
      // Not enough words, try smaller n-grams
      for (int n = targetN - 1; n >= 2; n--) {
        final ctxSize = n - 1;
        if (wordCount >= ctxSize) {
          return await _getSuggestionsForContext(
            completeWords.takeLast(ctxSize),
            n,
            topK,
          );
        }
      }
      // Fallback to unigrams
      await _ensureFreqLoaded();
      return _freqTopSorted!.take(topK).toList(growable: false);
    }

    // Try target n-gram first, then fallback to smaller n-grams if no results
    for (int n = targetN; n >= 2; n--) {
      final ctxSize = n - 1;
      if (wordCount >= ctxSize) {
        final result = await _getSuggestionsForContext(
          completeWords.takeLast(ctxSize),
          n,
          topK,
        );
        if (result.isNotEmpty) {
          return result;
        }
      }
    }

    // Fallback to most frequent unigrams if no n-gram matches
    await _ensureFreqLoaded();
    return _freqTopSorted!.take(topK).toList(growable: false);
  }

  /// Get suggestions for a specific context using n-gram probabilities.
  /// Tries multiple lookup formats (direct map, nested, flat prefix scan).
  Future<List<MapEntry<String, double>>> _getSuggestionsForContext(
    List<String> contextTokens,
    int n,
    int topK,
  ) async {
    final probMap = await _safeLoadProb(n);
    if (probMap == null || probMap.isEmpty) {
      return const [];
    }

    // Two possible formats:
    // A) by words: { "w1 w2 w3": { "next": p, ... } }
    // B) by words: { "w1 w2 w3 next": p, ... } (flat)
    // C) by ids:   { "1 2 3": { "4": p, ... } } or flat "1 2 3 4": p
    final suggestions = <MapEntry<String, double>>[];

    // Try both id and word keys to be safe
    final idKey = _toIdSequence(contextTokens);
    final wordKey = contextTokens.join(' ');
    final keysToTry = <_KeyFmt>[
      _KeyFmt(key: idKey, ids: true),
      _KeyFmt(key: wordKey, ids: false),
    ];

    for (final kfmt in keysToTry) {
      // 1) Direct key map: { contextKey: { next: p } }
      final direct = probMap[kfmt.key];
      if (direct is Map) {
        direct.forEach((k, v) {
          final p = _asDouble(v);
          if (p == null) return;
          final nextWord = _ensureWord(k.toString());
          if (nextWord != null) suggestions.add(MapEntry(nextWord, p));
        });
      }

      // 2) Nested descent: { id1: { id2: { ... { nextId: p } } } }
      final nested = _descendNested(probMap, contextTokens, useIds: kfmt.ids);
      if (nested is Map) {
        nested.forEach((k, v) {
          final p = _asDouble(v);
          if (p == null) return;
          final nextWord = _ensureWord(k.toString());
          if (nextWord != null) suggestions.add(MapEntry(nextWord, p));
        });
      }

      // 3) Flat prefix scan: { 'word1 word2': p } for 2-gram, { 'word1 word2 word3': p } for 3-gram, etc.
      // Format: "context next" where context is n-1 words and next is 1 word
      // For 2-gram with context "from": look for "from the" → extract "the"
      // For 3-gram with context "from the": look for "from the likes" → extract "likes"
      final prefix = '${kfmt.key} ';
      for (final key in probMap.keys.cast<String>()) {
        // Check if key starts with our context prefix (with space)
        if (!key.startsWith(prefix)) continue;

        // Extract everything after the context prefix
        final remainder = key.substring(prefix.length).trim();
        if (remainder.isEmpty) continue;

        // The remainder should be the next word(s), but we only want the immediate next word
        // For n-grams, the key format is exactly "context nextword" for the immediate next
        // So we can split by space and take the first word, or just use the whole remainder
        // if it's a single word (which it should be for proper n-gram format)
        final nextParts = remainder.split(' ');
        final nextIdOrWord =
            nextParts.first; // Take the first word after context

        final p = _asDouble(probMap[key]);
        if (p == null) continue;
        final nextWord = _ensureWord(nextIdOrWord);
        if (nextWord != null) {
          // Avoid duplicates - keep the highest probability
          final existingIndex = suggestions.indexWhere(
            (e) => e.key == nextWord,
          );
          if (existingIndex >= 0) {
            if (suggestions[existingIndex].value < p) {
              suggestions[existingIndex] = MapEntry(nextWord, p);
            }
          } else {
            suggestions.add(MapEntry(nextWord, p));
          }
        }
      }
    }

    if (suggestions.isNotEmpty) {
      // If using log probabilities, convert to probabilities and normalize per context
      if (_usesLogProba && suggestions.isNotEmpty) {
        // Find max logproba for this context to avoid overflow
        double? maxLogProb;
        for (final entry in suggestions) {
          if (maxLogProb == null || entry.value > maxLogProb) {
            maxLogProb = entry.value;
          }
        }

        // Convert logproba to probabilities and normalize
        if (maxLogProb != null) {
          double sum = 0.0;
          final tempProbs = <String, double>{};

          for (final entry in suggestions) {
            final diff = (entry.value - maxLogProb).clamp(-100, 100);
            final prob = math.exp(diff);
            tempProbs[entry.key] = prob;
            sum += prob;
          }

          // Normalize and update suggestions
          if (sum > 0) {
            suggestions.clear();
            tempProbs.forEach((word, prob) {
              suggestions.add(MapEntry(word, prob / sum));
            });
          }
        }
      }

      // Sort by probability descending (like Python's reverse=True)
      suggestions.sort((a, b) => b.value.compareTo(a.value));
      // Return top K suggestions (like Python's [:3])
      return suggestions.take(topK).toList(growable: false);
    }

    // No suggestions found for this n-gram
    return const [];
  }

  Future<Map<String, dynamic>?> _safeLoadProb(int n) async {
    try {
      return await _loadProbN(n);
    } catch (_) {
      return null;
    }
  }

  static List<String> _tokenize(String text) {
    final normalized = text.trim();
    if (normalized.isEmpty) return const [];
    // Simple whitespace tokenization, preserving casing for languages as-is
    return normalized.split(RegExp(r'\s+'));
  }

  // Normalization: lowercase for English/French/Kabyle; keep Arabic as-is. Strip simple punctuation.
  static List<String> _normalizeAndTokenize(String text, NgramLanguage lang) {
    String s = text;
    if (lang != NgramLanguage.arabic) {
      s = s.toLowerCase();
    }
    // Remove common punctuation around words to better match vocab
    s = s.replaceAll(RegExp(r'[\.,!?:;\-–—\(\)\[\]\{\}]'), ' ');
    s = s.replaceAll('"', ' ').replaceAll("'", ' ');
    final tokens = _tokenize(s);
    return tokens.where((t) => t.isNotEmpty).toList();
  }

  bool _isKnownWord(String word) {
    // If we have vocab, check presence; else assume known
    if (_vocab == null || _vocab!.isEmpty) return true;
    return _vocab!.containsKey(word);
  }

  String _toIdSequence(List<String> words) {
    if (_vocab == null || _vocab!.isEmpty) return words.join(' ');
    final ids = <String>[];
    for (final w in words) {
      final id = _vocab![w]?.toString();
      if (id == null) return words.join(' ');
      ids.add(id);
    }
    return ids.join(' ');
  }

  String? _idToWord(String id) {
    if (_vocabRev == null || _vocabRev!.isEmpty) return null;
    final w = _vocabRev![id];
    return w?.toString();
  }

  /// Always try to convert to word if it looks like a numeric ID.
  /// Returns the word if found, otherwise returns the original string.
  String? _ensureWord(String candidate) {
    // If it's purely numeric (like "14", "101", "102"), try to convert
    if (RegExp(r'^\d+$').hasMatch(candidate)) {
      final word = _idToWord(candidate);
      if (word != null && word.isNotEmpty) return word;
    }
    // If it's not numeric or conversion failed, return as-is (might already be a word)
    return candidate.isEmpty ? null : candidate;
  }

  /// If the probability file uses nested maps per token, walk the
  /// context sequence and return the final map of next-token -> prob.
  /// Returns null if structure doesn't match.
  Map<dynamic, dynamic>? _descendNested(
    Map<String, dynamic> root,
    List<String> contextTokens, {
    required bool useIds,
  }) {
    dynamic node = root;
    for (final token in contextTokens) {
      if (node is! Map) return null;
      final key = useIds ? (_vocab?[token]?.toString() ?? token) : token;
      final next = node[key];
      if (next == null) return null;
      node = next;
    }
    return node is Map ? node : null;
  }
}

extension _TakeLast<T> on List<T> {
  List<T> takeLast(int count) {
    if (count <= 0) return const [];
    if (length <= count) return List<T>.from(this);
    return sublist(length - count);
  }
}

double? _asDouble(Object? v) {
  if (v is num) return v.toDouble();
  if (v is String) {
    final parsed = double.tryParse(v);
    return parsed;
  }
  return null;
}

class _KeyFmt {
  const _KeyFmt({required this.key, required this.ids});
  final String key;
  final bool ids;
}

// helpers moved into class
