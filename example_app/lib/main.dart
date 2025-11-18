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
    const String modelPath = "/path/to/your/model.gguf";
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

  void _generateText() {
    if (!_isModelLoaded) {
      setState(() {
        _generatedText = "Please load a model first.";
      });
      return;
    }
    setState(() {
      _generatedText = "Generating...";
    });
    try {
      final String result = _llm.generate(_promptController.text, 50); // Generate up to 50 tokens
      setState(() {
        _generatedText = result;
      });
    } catch (e) {
      setState(() {
        _generatedText = "Error generating text: $e";
      });
    }
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
              onPressed: _loadModel,
              child: const Text('Load Model'),
            ),
            const SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: _generateText,
              child: const Text('Generate Text'),
            ),
            const SizedBox(height: 16.0),
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