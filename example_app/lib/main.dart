import 'package:flutter/material.dart';
import 'package:happy_phone_llm_flutter/happy_phone_llm_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Happy Phone LLM Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Happy Phone LLM Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final HappyPhoneLlm _llm = HappyPhoneLlm();
  final TextEditingController _promptController = TextEditingController();
  String _generatedText = "No text generated yet.";
  bool _isModelLoaded = false;
  bool _isGenerating = false; // New state variable for loading indicator

  @override
  void initState() {
    super.initState();
    _llm.createLlm();
  }

  @override
  void dispose() {
    _llm.destroyLlm();
    _promptController.dispose();
    super.dispose();
  }

  void _loadModel() {
    // For now, we'll use a dummy path. In a real app, this would be a path to a GGUF file.
    const String modelPath = "/data/local/tmp/qwen3-0.6b.gguf";
    try {
      _isModelLoaded = _llm.loadModel(modelPath);
      setState(() {
        _generatedText = _isModelLoaded ? "Model loaded successfully!" : "Failed to load model.";
      });
    } catch (e) {
      setState(() {
        _generatedText = "Error loading model: $e";
      });
    }
  }

  Future<void> _generateText() async { // Made asynchronous
    if (!_isModelLoaded) {
      setState(() {
        _generatedText = "Please load a model first.";
      });
      return;
    }
    setState(() {
      _generatedText = "Generating...";
      _isGenerating = true; // Set loading state to true
    });
    try {
      final String result = await _llm.generate(_promptController.text, 50); // Generate up to 50 tokens
      setState(() {
        _generatedText = result;
      });
    } catch (e) {
      setState(() {
        _generatedText = "Error generating text: $e";
      });
    } finally {
      setState(() {
        _isGenerating = false; // Set loading state to false
      });
    }
  }

  void _clearText() {
    _promptController.clear();
    setState(() {
      _generatedText = "No text generated yet.";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _promptController,
              decoration: const InputDecoration(
                labelText: 'Enter your prompt',
                border: OutlineInputBorder(),
              ),
              maxLines: 3,
            ),
            const SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: _isGenerating ? null : _loadModel, // Disable button while generating
              child: const Text('Load Model'),
            ),
            const SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: _isGenerating ? null : _generateText, // Disable button while generating
              child: const Text('Generate Text'),
            ),
            const SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: _isGenerating ? null : _clearText, // Disable button while generating
              child: const Text('Clear'),
            ),
            const SizedBox(height: 16.0),
            if (_isGenerating) // Show loading indicator when generating
              const Center(child: CircularProgressIndicator())
            else
              const Text(
                'Generated Text:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            const SizedBox(height: 8.0),
            Expanded(
              child: SingleChildScrollView(
                child: Text(
                  _generatedText,
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}