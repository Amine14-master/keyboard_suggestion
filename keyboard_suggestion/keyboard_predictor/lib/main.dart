import 'package:flutter/material.dart';

import 'services/ngram_predictor.dart';

void main() {
  runApp(const KeyboardPredictorApp());
}

class KeyboardPredictorApp extends StatelessWidget {
  const KeyboardPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'N-gram Keyboard Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const PredictorHomePage(),
    );
  }
}

class PredictorHomePage extends StatefulWidget {
  const PredictorHomePage({super.key});

  @override
  State<PredictorHomePage> createState() => _PredictorHomePageState();
}

class _PredictorHomePageState extends State<PredictorHomePage> {
  final TextEditingController _controller = TextEditingController();
  NgramLanguage _language = NgramLanguage.english;
  NgramPredictor? _predictor;
  List<MapEntry<String, double>> _suggestions = const [];
  bool _loading = false;
  String _lastProcessedText = '';

  @override
  void initState() {
    super.initState();
    _predictor = NgramPredictor(_language);
    // Use both listener and onChanged to ensure updates happen
    _controller.addListener(_onTextChanged);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _onTextChanged() async {
    final text = _controller.text;
    if (_predictor == null) return;

    // Always update suggestions when text changes
    // Force update if text ends with space (new word completed)
    final endsWithSpace = text.endsWith(' ');
    final textChanged = text != _lastProcessedText;

    if (!textChanged && !endsWithSpace) {
      return; // Skip if text hasn't changed and doesn't end with space
    }

    _lastProcessedText = text;

    if (mounted) {
      setState(() {
        _loading = true;
      });
    }

    try {
      final res = await _predictor!.suggest(text, topK: 3);
      if (mounted) {
        setState(() {
          _suggestions = res;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _suggestions = const [];
          _loading = false;
        });
      }
    }
  }

  void _onSuggestionTap(String word) {
    final text = _controller.text.trimRight();
    final newText = text.isEmpty ? word : '$text $word';
    // Update last processed to current text first to avoid double processing
    _lastProcessedText = _controller.text;
    _controller.text = newText;
    _controller.selection = TextSelection.fromPosition(
      TextPosition(offset: _controller.text.length),
    );
    // The listener will automatically trigger _onTextChanged
  }

  void _onLanguageChanged(NgramLanguage lang) {
    setState(() {
      _language = lang;
      _predictor = NgramPredictor(_language);
      _suggestions = const [];
    });
    _onTextChanged();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Next-word Suggestions (1-5 grams)'),
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: _LanguageSelector(
              value: _language,
              onChanged: _onLanguageChanged,
            ),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _controller,
              maxLines: null,
              decoration: const InputDecoration(
                labelText: 'Type here',
                border: OutlineInputBorder(),
              ),
              // onChanged handled by controller listener
            ),
            const SizedBox(height: 12),
            if (_loading) const LinearProgressIndicator(minHeight: 2),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: _suggestions
                  .map(
                    (e) => ActionChip(
                      label: Text('${e.key} (${e.value.toStringAsFixed(3)})'),
                      onPressed: () => _onSuggestionTap(e.key),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 8),
            if (!_loading && _suggestions.isEmpty)
              const Text(
                'No suggestions yet. Start typing to see predictions.',
                textAlign: TextAlign.center,
              ),
          ],
        ),
      ),
    );
  }
}

class _LanguageSelector extends StatelessWidget {
  const _LanguageSelector({required this.value, required this.onChanged});

  final NgramLanguage value;
  final ValueChanged<NgramLanguage> onChanged;

  @override
  Widget build(BuildContext context) {
    return DropdownButton<NgramLanguage>(
      value: value,
      onChanged: (v) {
        if (v != null) onChanged(v);
      },
      items: const [
        DropdownMenuItem(value: NgramLanguage.english, child: Text('English')),
        DropdownMenuItem(value: NgramLanguage.french, child: Text('French')),
        DropdownMenuItem(value: NgramLanguage.arabic, child: Text('Arabic')),
        DropdownMenuItem(
          value: NgramLanguage.kabyle,
          child: Text('Kabyle (Amazigh)'),
        ),
      ],
    );
  }
}
