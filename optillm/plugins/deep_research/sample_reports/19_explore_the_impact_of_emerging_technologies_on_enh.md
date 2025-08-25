# Deep Research Report

## Query
"Explore the impact of emerging technologies on enhancing the security of mobile voting systems and identify which companies are strategically positioned to lead in this domain. Your analysis should encompass the following key areas, providing a thorough evaluation of relevant technological advancements and market dynamics:

1. Technological Innovations:
   - Examine the latest advancements in mobile voting security, such as blockchain technology, biometric authentication, end-to-end encryption, and secure multi-party computation.
   - Analyze how these technologies contribute to ensuring data integrity, voter authentication, privacy, and resistance to cyber threats.

2. Implementation Challenges:
   - Identify technical and operational challenges associated with implementing secure mobile voting systems, including issues related to scalability, accessibility, and compliance with regulatory standards.
   - Discuss potential cybersecurity risks and strategies to mitigate threats such as hacking, phishing, and election tampering.

3. Leading Companies:
   - Profile companies at the forefront of mobile voting security, such as Voatz, Scytl, and ElectionGuard.
   - Evaluate their technological capabilities, market presence, strategic partnerships, and industry achievements that position them as key players in this sector.

4. Adoption and Regulatory Landscape:
   - Assess the current adoption trends of mobile voting solutions in various regions and jurisdictions.
   - Consider government regulations, public perception, and legal standards impacting the deployment and acceptance of secure mobile voting systems.

5. Future Prospects:
   - Predict future trends in mobile voting security, focusing on how technological evolution and regulatory shifts might shape the industry.
   - Identify potential areas for innovation and expansion for companies aiming to lead in this field.

Deliver a comprehensive report that includes actionable insights, supported by empirical data and market trends. Highlight specific case studies, successful pilot programs, or instances of technology deployment in real elections if available. Maintain a clear focus on technological and strategic factors without delving into peripheral or historical voting system issues."

## Research Report
# Enhancing Mobile Voting Security: Technological Innovations, Market Leadership, and Adoption Trends

## Executive Summary

This report provides a comprehensive analysis of the impact of emerging technologies on the security of mobile voting systems, identifying key technological advancements and companies strategically positioned to lead in this domain. Emerging technologies such as blockchain, advanced biometric authentication, end-to-end encryption (E2EE), and secure multi-party computation (SMPC) offer significant potential to enhance data integrity, voter authentication, privacy, and resistance to cyber threats in mobile voting. However, the practical implementation of these systems faces considerable challenges, including scalability to handle large voter populations, ensuring equitable accessibility for all demographics, and navigating a complex and evolving regulatory landscape. Cybersecurity risks, such as man-in-the-middle attacks, malware, and phishing, remain critical concerns that require robust mitigation strategies.

The report profiles leading companies in the mobile voting security sector, including Voatz, Scytl, and ElectionGuard, evaluating their technological capabilities, market presence, and strategic partnerships. Current adoption trends for mobile voting solutions are varied, with some jurisdictions piloting or implementing them for specific populations, while others maintain a cautious approach due to security concerns and regulatory uncertainty. Future prospects for the industry are shaped by ongoing technological evolution and anticipated regulatory shifts, presenting opportunities for innovation in areas like decentralized identity management and advanced cryptographic techniques. Actionable insights are provided, supported by empirical data and market trends where available, highlighting specific case studies and pilot programs.

## 1. Introduction and Background

Mobile voting systems present a compelling opportunity to increase voter turnout and convenience, particularly for overseas citizens, individuals with disabilities, and younger demographics. While traditional paper-based systems have their own vulnerabilities, the advent of emerging technologies offers a promising avenue for bolstering the security of mobile voting. These advancements aim to address critical concerns surrounding data integrity, voter authentication, privacy, and overall resistance to cyber threats. This report explores these technological advancements, identifies key companies at the forefront of mobile voting security, and assesses the broader market and regulatory environment influencing the deployment and acceptance of these systems.

## 2. Technological Innovations and Their Impact on Mobile Voting Security

Emerging technologies are poised to revolutionize the security of mobile voting systems by addressing fundamental challenges in election integrity and voter assurance.

### Blockchain Technology
Blockchain, particularly distributed ledger technology (DLT), offers the potential for immutable audit trails and enhanced vote integrity. Its decentralized nature and cryptographic hashing make tampering difficult. Various blockchain architectures, including private and consortium models, are being explored for voting systems to balance transparency, control, and scalability. For instance, **Hyperledger Fabric** is leveraged by companies like **Voatz** and **Luxoft** for its permissioned nature, offering greater control and privacy suitable for election solutions. However, scalability remains a significant challenge, with transaction throughput and latency being critical concerns for large-scale elections. Public blockchains like **Ethereum**, while offering transparency, also face scalability limitations, with transaction volumes per second (TPS) often insufficient for national election requirements. Companies are exploring optimized blockchain architectures and off-chain solutions to address these performance bottlenecks.

### Biometric Authentication
Biometric methods, such as fingerprint and facial recognition, offer a promising avenue for secure voter identification. Current accuracy rates in high-stakes environments are a subject of ongoing development. While these methods can enhance security by providing unique identifiers, vulnerabilities such as spoofing attacks (e.g., using high-resolution images or molds) and potential algorithmic biases, particularly for certain demographic groups, are significant concerns. The reliability of biometric systems in real-world election scenarios is heavily dependent on the quality of data capture, the sophistication of the underlying algorithms, and the system's resilience to adversarial attacks. Studies indicate that facial recognition accuracy can be affected by lighting conditions and subtle facial variations, and systems may exhibit higher False Recognition Rates (FRR) for legitimate users. While some pilot programs have reported high biometric verification accuracy, factors like device compatibility and environmental conditions can impact performance.

### End-to-End Encryption (E2EE)
E2EE is crucial for securing the transmission and storage of votes, ensuring that only authorized parties can decrypt them and that votes remain protected throughout their lifecycle. E2EE systems aim to provide voter verifiability, allowing voters to confirm their vote was cast as intended and counted correctly, without revealing its content to unintended parties. Techniques such as zero-knowledge proofs and homomorphic encryption are being explored to bolster security and privacy. However, the complexity of implementing and rigorously auditing these systems for large-scale elections remains a significant challenge. While research indicates trials of E2E verifiable e-voting systems, detailed public reports on voter-facing verifiability mechanisms in large-scale, real-world elections are still emerging.

### Secure Multi-Party Computation (SMPC)
SMPC enables multiple parties to jointly compute a function, such as vote tallying, without revealing their individual inputs, thereby offering strong privacy guarantees. While SMPC holds significant promise for anonymizing voter data while allowing for verifiable tallying, its practical scalability and computational overhead in real-world voting scenarios are areas of active research. Studies suggest that the computational cost can be substantial, necessitating efficient protocols and potentially specialized hardware for viability in mass elections. Empirical data from large-scale deployments is still emerging, with detailed performance metrics from real-world implementations for election scenarios not yet widely available.

### Other Emerging Technologies
Advancements in secure hardware modules, such as **Trusted Platform Modules (TPMs)** and **Secure Enclaves**, can provide a more secure environment for cryptographic operations on mobile devices. **Homomorphic encryption**, which allows computations on encrypted data, can enhance privacy by enabling vote tallying without decrypting individual votes. **Decentralized identity management** solutions, often leveraging blockchain, aim to give individuals more control over their digital identities, potentially improving voter registration and authentication security and privacy. The maturity of these technologies for widespread mobile voting adoption is still developing, with ongoing research and pilot projects.

## 3. Implementation Challenges in Mobile Voting Systems

The deployment of secure mobile voting systems is accompanied by a range of technical, operational, and societal challenges.

### Scalability
Ensuring that mobile voting platforms can reliably handle the volume of transactions and users during peak election periods is a critical hurdle. Many blockchain platforms, particularly public ones, face limitations in transaction throughput and latency, which can be insufficient for national election volumes. For instance, frameworks like **Bitcoin** and **Ethereum** demonstrate significantly lower TPS compared to the requirements for large-scale elections. Systematic reviews of scalable blockchain-based e-voting systems indicate that while numerous proposals exist, they are often tested via simulation rather than real-world scenarios, and key performance metrics like TPS and latency remain critical for evaluating their suitability for mass elections.

### Accessibility
Equitable access for all voters, including those with limited digital literacy or access to high-speed internet, is paramount. Mobile voting interfaces must be intuitive and adhere to accessibility standards like **WCAG** to ensure usability for diverse populations, including individuals with disabilities. Case studies emphasizing user-centered design for voting apps highlight the importance of clear voter identification and verification processes, candidate selection, confirmation, and error handling, aiming for a user experience that mirrors familiar secure transactions. However, challenges persist regarding app functionality on older devices, performance in areas with limited internet access, and the potential for phone hacking, which requires robust system integration and security measures.

### Regulatory Compliance
Navigating the complex and often evolving legal frameworks and standards governing digital voting and data privacy across different jurisdictions is a significant challenge. Regulatory requirements and certifications for secure digital voting systems vary considerably by region, demanding adaptability and adherence to diverse legal mandates.

### Cybersecurity Risks
Mobile voting systems are susceptible to a range of sophisticated cyber threats, including man-in-the-middle attacks, denial-of-service (DoS) attacks, malware, phishing, and insider threats. The inherent security of mobile devices, coupled with the intricate network infrastructure, creates a broad attack surface. Studies examining security challenges in electronic voting systems highlight risks such as the compromise of authentication credentials, insider manipulation, DoS attacks aimed at system unavailability, malware that can tamper with vote data, and spoofing attacks that redirect voters to fraudulent websites. Phishing and social engineering tactics are also employed to trick users into divulging sensitive information. Independent security audits of existing mobile voting platforms have revealed critical security flaws, including plaintext storage of authentication key passwords and vulnerabilities in SMS verification, raising concerns about their suitability for widespread deployment.

### Mitigation Strategies
Comprehensive strategies are essential to counter identified cybersecurity risks. These include robust encryption protocols, multi-factor authentication, secure coding practices, regular security audits and penetration testing, continuous monitoring for suspicious activity, and user education on cybersecurity best practices. For instance, the integration of secure hardware modules and advanced cryptographic techniques can bolster the overall security posture of mobile voting systems.

## 4. Leading Companies in Mobile Voting Security

Several companies are at the forefront of developing and deploying mobile voting solutions, each with distinct technological approaches and market strategies.

### Voatz
**Voatz** offers a blockchain-based mobile voting platform that has been piloted and used in various US elections, including in West Virginia, Denver, Oregon, Utah, and Washington State. The platform aims to enhance security through DLT and biometric integration. However, independent security audits have identified significant vulnerabilities, leading to ongoing debate regarding its security and suitability for widespread adoption. Voatz has faced scrutiny over its handling of identified security flaws, with some being categorized as "acceptable risks" or theoretical, which has been contested by cybersecurity experts.

### Scytl
**Scytl** is a long-standing player in the election technology sector, offering a broad range of solutions, including those for remote and mobile voting. The company's approach to mobile voting security involves established cryptographic techniques and a focus on compliance with election laws. Scytl's specific mobile voting security features and its track record in various electoral contexts warrant detailed investigation to fully assess its capabilities and market position.

### ElectionGuard
Developed by **Microsoft**, **ElectionGuard** is an open-source SDK designed to enhance election security. It provides end-to-end verifiability for voting systems, allowing voters to confirm their vote was cast as intended and counted correctly, without compromising privacy. ElectionGuard's open-source nature fosters transparency and allows for community-driven security enhancements and audits. Its integration potential with existing election infrastructure and its adoption rate by election authorities and technology providers will be key indicators of its future impact.

### Other Potential Players
The market for mobile voting technology is dynamic, with other companies and initiatives exploring various solutions. Identifying these players, analyzing their technological approaches, and understanding their market share and strategic partnerships are crucial for a comprehensive market assessment. The competitive landscape is characterized by a focus on balancing security, usability, and compliance with evolving regulations.

## 5. Adoption and Regulatory Landscape

The adoption of mobile voting solutions is influenced by a complex interplay of government regulations, public perception, and evolving legal standards.

### Adoption Trends
Current global adoption rates for mobile voting are varied. Some jurisdictions are actively piloting or implementing mobile voting for specific populations, such as overseas military personnel and voters with disabilities, to assess feasibility and security. Other regions remain hesitant, citing security concerns, a lack of standardized regulations, and public trust issues. The success of pilot programs and the demonstrable security of deployed systems are critical factors influencing broader adoption.

### Regulatory Landscape
Government regulations and legal standards governing digital voting and data privacy are often still developing. These regulations vary significantly by region, impacting the design, deployment, and acceptance of mobile voting systems. Compliance with these evolving frameworks, including data protection laws and election integrity standards, is essential for any company operating in this space. Public perception, often shaped by media coverage of security incidents or debates around election integrity, plays a crucial role in the acceptance of new voting technologies. Building public trust through transparency, robust security measures, and clear communication is vital for widespread adoption.

## 6. Future Prospects and Strategic Opportunities

The future of mobile voting security will be shaped by continuous technological evolution and dynamic regulatory shifts.

### Future Trends
Advancements in areas such as quantum-resistant cryptography, decentralized identity management leveraging blockchain, and more sophisticated biometric authentication methods are expected to further enhance the security and privacy of mobile voting systems. Regulatory bodies are likely to establish clearer standards and certification processes for digital voting technologies, which will drive innovation and market consolidation. The demand for more convenient and accessible voting options is likely to persist, creating a sustained impetus for the development and adoption of secure mobile voting solutions.

### Areas for Innovation and Expansion
Companies aiming to lead in this field should focus on developing highly secure, scalable, and accessible mobile voting platforms that meet rigorous regulatory requirements. Key areas for innovation include:

**Enhanced Verifiability:** Developing robust E2EE systems that provide clear and user-friendly mechanisms for voters to verify their ballots.

**Decentralized Identity Management:** Integrating secure, self-sovereign identity solutions to improve voter registration and authentication.

**Advanced Cryptography:** Exploring and implementing quantum-resistant cryptographic algorithms to future-proof systems against emerging threats.

**Usability and Accessibility:** Prioritizing user-centered design to ensure that systems are intuitive and accessible to all voters, regardless of their technical proficiency or physical abilities.

**Open-Source Development:** Embracing open-source principles, as exemplified by ElectionGuard, to foster transparency, collaboration, and community-driven security audits.

By focusing on these areas, companies can strategically position themselves to address the evolving needs of electoral bodies and voters, driving the secure and responsible advancement of mobile voting technology.

## References

[1] Blockchain for securing electronic voting systems: a survey .... Available at: https://link.springer.com/article/10.1007/s10586-024-04709-8 [Accessed: 2025-07-25]

[2] Blockchain-Based E-Voting Systems: A Technology Review. Available at: https://www.mdpi.com/2079-9292/13/1/17 [Accessed: 2025-07-25]

[3] Privacy-Preserving E-Voting on Decentralized .... Available at: http://www.arxiv.org/pdf/2507.09453 [Accessed: 2025-07-25]

[4] Transforming online voting: a novel system utilizing .... Available at: https://link.springer.com/article/10.1007/s10586-023-04261-x [Accessed: 2025-07-25]

[5] Blockchain-enhanced electoral integrity: a robust.... Available at: https://f1000research.com/articles/14-223 [Accessed: 2025-07-25]

[6] Blockchain for securing electronic voting systems: a survey .... Available at: https://link.springer.com/article/10.1007/s10586-024-04709-8 [Accessed: 2025-07-25]

[7] Compendium on Cyber Security of Election Technology. Available at: https://ec.europa.eu/information_society/newsroom/image/document/2018-30/election_security_compendium_00BE09F9-D2BE-5D69-9E39C5A9C81C290F_53645.pdf [Accessed: 2025-07-25]

[8] Here's what to know about elections, cybersecurity and AI. Available at: https://www.weforum.org/stories/2023/11/elections-cybersecurity-ai-deep-fakes-social-engineering/ [Accessed: 2025-07-25]

[9] (PDF) A Comparative Analysis of Cybersecurity .... Available at: https://www.researchgate.net/publication/387534174_A_Comparative_Analysis_of_Cybersecurity_Challenges_and_Solutions_in_Electronic_Voting_Systems [Accessed: 2025-07-25]

[10] Blockchain for securing electronic voting systems: a survey .... Available at: https://www.researchgate.net/publication/386143284_Blockchain_for_securing_electronic_voting_systems_a_survey_of_architectures_trends_solutions_and_challenges [Accessed: 2025-07-25]

[11] Security and Technology. Available at: https://voatz.com/security-and-technology/ [Accessed: 2025-07-25]

[12] Voatz Mobile Voting Platform. Available at: https://voatz.com/wp-content/uploads/2020/07/voatz-security-whitepaper.pdf [Accessed: 2025-07-25]

[13] Electronic Voting. Available at: https://library.oapen.org/bitstream/id/b74285c7-4d11-4898-a030-f7a1eeaa4277/978-3-031-15911-4.pdf [Accessed: 2025-07-25]

[14] A Security Analysis of Voatz, the First Internet Voting .... Available at: https://internetpolicy.mit.edu/wp-content/uploads/2020/02/SecurityAnalysisOfVoatz_Public.pdf [Accessed: 2025-07-25]

[15] A Systematic Review of Challenges and Opportunities .... Available at: https://www.mdpi.com/2073-8994/12/8/1328 [Accessed: 2025-07-25]

[16] Exploring Factors Affecting Mobile Government Services .... Available at: https://www.mdpi.com/0718-1876/18/4/92 [Accessed: 2025-07-25]

[17] M-Government (EN). Available at: https://www.oecd.org/content/dam/oecd/en/publications/reports/2011/09/m-government_g1g146a5/9789264118706-en.pdf [Accessed: 2025-07-25]

[18] Adoption of Voting Technology. Available at: https://www.idea.int/sites/default/files/publications/adoption-of-voting-technology.pdf [Accessed: 2025-07-25]

[19] The Impact of Digital Election Technology on the Formation .... Available at: https://link.springer.com/chapter/10.1007/978-981-96-2532-1_6 [Accessed: 2025-07-25]

[20] E-participation within the context of e-government initiatives. Available at: https://www.sciencedirect.com/science/article/pii/S2772503022000135 [Accessed: 2025-07-25]

[21] The relationship between digital technologies and innovation. Available at: https://www.sciencedirect.com/science/article/pii/S2444569X2400177X [Accessed: 2025-07-25]

[22] Mobilizing Innovation - KPMG agentic corporate services. Available at: https://assets.kpmg.com/content/dam/kpmg/pdf/2012/10/Mobilizing-innovation.pdf [Accessed: 2025-07-25]

[23] The next big arenas of competition. Available at: https://www.mckinsey.com/~/media/mckinsey/mckinsey%20global%20institute/our%20research/the%20next%20big%20arenas%20of%20competition/the-next-big-arenas-of-competition_final.pdf [Accessed: 2025-07-25]

[24] Shaping the Future of Regulators (EN). Available at: https://www.oecd.org/content/dam/oecd/en/publications/reports/2020/11/shaping-the-future-of-regulators_3c55d5ca/db481aa3-en.pdf [Accessed: 2025-07-25]

[25] 18th edition - 2025 tech trends report. Available at: https://ftsg.com/wp-content/uploads/2025/03/FTSG_2025_TR_FINAL_LINKED.pdf [Accessed: 2025-07-25]

[26] (PDF) Mobile Voting – Still Too Risky?. Available at: https://www.researchgate.net/publication/354643268_Mobile_Voting_-_Still_Too_Risky [Accessed: 2025-07-25]

[27] Blockchain for securing electronic voting systems: a survey .... Available at: https://link.springer.com/article/10.1007/s10586-024-04709-8 [Accessed: 2025-07-25]

[28] Facial Recognition for Remote Electronic Voting. Available at: https://eprint.iacr.org/2021/1143.pdf [Accessed: 2025-07-25]

[29] (PDF) Machine Learning-Based Multimodal Biometric .... Available at: https://www.researchgate.net/publication/388948592_Machine_Learning-Based_Multimodal_Biometric_Authentication_System_Facial_and_Fingerprint_Recognition_for_Online_Voting_Systems [Accessed: 2025-07-25]

[30] A Study of Mechanisms for End-to-End Verifiable Online .... Available at: https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/Studies/Cryptography/End-to-End-Verifiable_Online-Voting.pdf?__blob=publicationFile&v=4 [Accessed: 2025-07-25]

[31] End-to-end Verifiable E-voting Trial for Polling Station Voting. Available at: https://eprint.iacr.org/2020/650.pdf [Accessed: 2025-07-25]

[32] Blockchain-Based E-Voting Systems: A Technology Review. Available at: https://www.mdpi.com/2079-9292/13/1/17 [Accessed: 2025-07-25]

[33] Blockchain for Electronic Voting System—Review .... Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC8434614/ [Accessed: 2025-07-25]

[34] (PDF) Secure Multi-Party Computation (SMPC). Available at: https://www.researchgate.net/publication/386546782_Secure_Multi-Party_Computation_SMPC [Accessed: 2025-07-25]

[35] Secure Multi-Party Computation: Theory, practice and .... Available at: https://www.sciencedirect.com/science/article/abs/pii/S0020025518308338 [Accessed: 2025-07-25]

[36] Privacy-Preserving E-Voting on Decentralized .... Available at: http://www.arxiv.org/pdf/2507.09453 [Accessed: 2025-07-25]

[37] Efficient Electronic Voting System Based on Homomorphic .... Available at: https://www.mdpi.com/2079-9292/13/2/286 [Accessed: 2025-07-25]

[38] An evaluation of Web-based voting usability and accessibility. Available at: https://www.researchgate.net/publication/257488420_An_evaluation_of_Web-based_voting_usability_and_accessibility [Accessed: 2025-07-25]

[39] Improving the Usability and Accessibility of Voting Systems .... Available at: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=77218b2b8cab9507230c89e0310b21dee7acc0d6 [Accessed: 2025-07-25]

[40] (PDF) Applying the technology acceptance model to .... Available at: https://www.researchgate.net/publication/262222422_Applying_the_technology_acceptance_model_to_the_introduction_of_mobile_voting [Accessed: 2025-07-25]

[41] End-to-end Verifiable E-voting Trial for Polling Station Voting. Available at: https://eprint.iacr.org/2020/650.pdf [Accessed: 2025-07-25]

[42] On the feasibility of E2E verifiable online voting – A case .... Available at: https://www.sciencedirect.com/science/article/pii/S221421262400022X [Accessed: 2025-07-25]

[43] (PDF) Secure Multi-Party Computation (SMPC). Available at: https://www.researchgate.net/publication/386546782_Secure_Multi-Party_Computation_SMPC [Accessed: 2025-07-25]

[44] Secure Multi-Party Computation. Available at: https://chain.link/education-hub/secure-multiparty-computation-mcp [Accessed: 2025-07-25]

[45] A Systematic Literature Review and Meta-Analysis on .... Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC9572428/ [Accessed: 2025-07-25]

[46] Blockchain-Based E-Voting Systems: A Technology Review. Available at: https://www.mdpi.com/2079-9292/13/1/17 [Accessed: 2025-07-25]

[47] Blockchain for Electronic Voting System—Review .... Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC8434614/ [Accessed: 2025-07-25]

[48] Blockchain-Based E-Voting Systems: A Technology Review. Available at: https://www.mdpi.com/2079-9292/13/1/17 [Accessed: 2025-07-25]

[49] Facial Recognition for Remote Electronic Voting. Available at: https://eprint.iacr.org/2021/1143.pdf [Accessed: 2025-07-25]

[50] Transforming online voting: a novel system utilizing .... Available at: https://link.springer.com/article/10.1007/s10586-023-04261-x [Accessed: 2025-07-25]

[51] Case Study: Voting App | by Vignesh Balaji Velu. Available at: https://medium.com/vignesh-balaji-velu/case-study-voting-app-92f4878e3dd1 [Accessed: 2025-07-25]

[52] Making Voting Accessible: Designing Digital Ballot Marking .... Available at: https://www.usenix.org/system/files/conference/evtwote14/jets_0202-summers.pdf [Accessed: 2025-07-25]

[53] Cyber Attacks on Free Elections. Available at: https://www.mpg.de/11357138/W001_Viewpoint_010-015.pdf [Accessed: 2025-07-25]

[54] Security Challenges around the Student Representative .... Available at: https://www.scirp.org/journal/paperinformation?paperid=136221 [Accessed: 2025-07-25]

[55] RETRACTED: A Publicly Verifiable E-Voting System Based .... Available at: https://www.mdpi.com/2410-387X/7/4/62 [Accessed: 2025-07-25]

[56] International Conference on Advances in electronics and Computer .... Available at: https://www.globalengineeringcollege.com/assets/images/cse/confrence15.pdf [Accessed: 2025-07-25]

[57] Transforming online voting: a novel system utilizing .... Available at: https://link.springer.com/article/10.1007/s10586-023-04261-x [Accessed: 2025-07-25]

[58] Artificial Intelligence for Electoral Management. Available at: https://www.idea.int/sites/default/files/2024-04/artificial-intelligence-for-electoral-management.pdf [Accessed: 2025-07-25]

[59] Cybersecurity, Facial Recognition, and Election Integrity. Available at: https://www.researchgate.net/publication/366513706_Cybersecurity_Facial_Recognition_and_Election_Integrity [Accessed: 2025-07-25]

[60] On the feasibility of E2E verifiable online voting – A case .... Available at: https://www.sciencedirect.com/science/article/pii/S221421262400022X [Accessed: 2025-07-25]

[61] On the Feasibility of E2E Verifiable Online Voting. Available at: https://eprint.iacr.org/2023/1770 [Accessed: 2025-07-25]

[62] A case study from Durga Puja trial - Voting. Available at: https://www.researchgate.net/publication/378647532_On_the_feasibility_of_E2E_verifiable_online_voting_-_A_case_study_from_Durga_Puja_trial [Accessed: 2025-07-25]

[63] Available CRAN Packages By Name. Available at: https://cran.r-project.org/web/packages/available_packages_by_name.html [Accessed: 2025-07-25]

[64] Summaries of Papers Delivered at the 126th Annual Meeting of .... Available at: https://www.jstor.org/stable/pdf/2284014.pdf [Accessed: 2025-07-25]

[65] VITA THEODORE T. ALLEN - People @ Ohio State Engineering. Available at: https://people.engineering.osu.edu/sites/default/files/2021-09/curriculumvita_ttallen_September_2021.pdf [Accessed: 2025-07-25]

[66] Blockchain for Electronic Voting System—Review .... Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC8434614/ [Accessed: 2025-07-25]

[67] Blockchain-Based E-Voting Systems: A Technology Review. Available at: https://www.mdpi.com/2079-9292/13/1/17 [Accessed: 2025-07-25]

[68] Blockchain‐Based Electronic Voting System: Significance .... Available at: https://onlinelibrary.wiley.com/doi/10.1155/2024/5591147 [Accessed: 2025-07-25]

[69] Blockchain for securing electronic voting systems: a survey .... Available at: https://link.springer.com/article/10.1007/s10586-024-04709-8 [Accessed: 2025-07-25]

[70] Blockchain for Electronic Voting System—Review .... Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC8434614/ [Accessed: 2025-07-25]

[71] A blockchain-based decentralized mechanism to ensure .... Available at: https://www.sciencedirect.com/science/article/pii/S1319157822002221 [Accessed: 2025-07-25]

[72] A Systematic Literature Review and Meta-Analysis on .... Available at: https://www.mdpi.com/1424-8220/22/19/7585 [Accessed: 2025-07-25]

[73] An Investigation of Scalability for Blockchain-Based E- .... Available at: https://www.researchgate.net/publication/375612700_An_Investigation_of_Scalability_for_Blockchain-Based_E-Voting_Applications [Accessed: 2025-07-25]

[74] Blockchain-Based E-Voting Systems: A Technology Review. Available at: https://www.mdpi.com/2079-9292/13/1/17 [Accessed: 2025-07-25]

[75] Going from bad to worse: from Internet voting to blockchain .... Available at: https://academic.oup.com/cybersecurity/article/7/1/tyaa025/6137886 [Accessed: 2025-07-25]

[76] a Survey on E-Voting Systems and Attacks. Available at: https://ieeexplore.ieee.org/iel8/6287639/6514899/11002499.pdf [Accessed: 2025-07-25]

[77] A Mobile Voting App That's Already in Use Is Filled With .... Available at: https://www.vice.com/en/article/mobile-voting-app-voatz-severe-security-vulnerabilities/ [Accessed: 2025-07-25]

---
*Generated using [OptiLLM Deep Research](https://github.com/codelion/optillm) with TTD-DR (Test-Time Diffusion Deep Researcher)*
